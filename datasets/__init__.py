from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
from PIL import Image
import os
import logging
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustImageFolder(ImageFolder):
    """增强版ImageFolder，自动处理损坏文件"""
    def __init__(self, root, transform=None):
        super().__init__(
            root=root,
            transform=transform,
            loader=self._safe_loader,
            is_valid_file=self._is_valid_file
        )
        # 重新构建有效的样本列表
        self.samples = [s for s in self.samples if self._is_valid_file(s[0])]
        self.imgs = self.samples
        logger.info(f"数据集初始化完成，有效样本数: {len(self.samples)}")

    def _is_valid_file(self, path):
        """验证文件有效性"""
        try:
            if not os.path.exists(path):
                logger.warning(f"文件不存在: {path}")
                return False
            if os.path.getsize(path) == 0:
                logger.warning(f"空文件: {path}")
                return False
            with open(path, 'rb') as f:
                Image.open(f).verify()
            return True
        except Exception as e:
            logger.warning(f"无效文件 {path}: {str(e)}")
            return False

    def _safe_loader(self, path):
        """安全加载器"""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()
                return img.convert('RGB')
        except Exception as e:
            logger.error(f"加载失败 {path}: {str(e)}")
            # 返回占位图像
            return Image.new('RGB', (256, 256), (0, 0, 0))

class SafeCompose(transforms.Compose):
    """安全数据增强管道"""
    def __call__(self, img):
        try:
            return super().__call__(img)
        except Exception as e:
            logger.error(f"数据增强失败: {str(e)}")
            return torch.zeros(3, 224, 224)  # 返回空白张量

def get_transforms(is_train=True):
    """获取数据增强管道"""
    if is_train:
        return SafeCompose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return SafeCompose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

def setup_dataloader(
    folder_path,
    batch_size=20,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
    is_train=True,
    is_distributed=None
):
    """
    创建数据加载器
    Args:
        is_distributed: 强制指定是否分布式模式(None时自动检测)
    """
    # 自动检测分布式模式
    if is_distributed is None:
        is_distributed = dist.is_initialized() if dist.is_available() else False

    dataset = RobustImageFolder(
        folder_path,
        transform=get_transforms(is_train)
    )

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset)
        logger.info(f"初始化分布式采样器 (rank={dist.get_rank()}, world_size={dist.get_world_size()})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: torch.utils.data.default_collate(
            [item for item in x if item[0] is not None]  # 过滤无效数据
        )
    )
    return loader

def setup_infinity_train_dataloader(
    folder_path,
    batch_size=20,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
    is_distributed=None
):
    """无限数据流加载器"""
    loader = setup_dataloader(
        folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        is_train=True,
        is_distributed=is_distributed
    )

    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        epoch += 1
        try:
            yield from loader
        except Exception as e:
            logger.error(f"数据流错误: {str(e)}")
            continue

def setup_val_dataloader(
    folder_path,
    batch_size=20,
    num_workers=16,
    pin_memory=True,
    drop_last=False,  # 验证集通常不drop_last
    is_distributed=None
):
    """验证集加载器"""
    return setup_dataloader(
        folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        is_train=False,
        is_distributed=is_distributed
    )
