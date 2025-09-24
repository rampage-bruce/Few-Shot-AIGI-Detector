from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
from PIL import Image
import os
import logging
import torch
import pickle
import hashlib
import time
from pathlib import Path

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



class FastImageFolder(ImageFolder):
    """Optimized ImageFolder with efficient caching and parallel validation"""
    # _global_cache = {}  # Class-level cache to share across instances

    def __init__(self, root, transform=None, cache_dir=".dataset_cache", skip_validation=False):
        self.root = root
        self.cache_dir = cache_dir
        print("cache directory:", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)


        # Generate a unique cache key based on dataset path and contents
        self.cache_key = self._generate_cache_key(root)
        self.cache_file = os.path.join(cache_dir, f"{self.cache_key}.pkl")

        # Try to load from global cache first (remove the global cache)
        # if self.cache_key in FastImageFolder._global_cache:
        #     self.valid_files = FastImageFolder._global_cache[self.cache_key]
        #     logger.info(f"Using global cache for {root} with {len(self.valid_files)} files")

        # Try to load from disk cache
        self.valid_files = self._load_cache()
        if self.valid_files is not None:
            logger.info(f"Loaded cache for {root} with {len(self.valid_files)} files")
            # FastImageFolder._global_cache[self.cache_key] = self.valid_files
        else:
            # Need to build cache
            if skip_validation:
                logger.warning("No cache found but skip_validation=True. This may cause errors.")
                self.valid_files = set()
            else:
                logger.info(f"Building cache for {root}. This may take a while...")
                self.valid_files = self._build_cache()
                self._save_cache()
                # FastImageFolder._global_cache[self.cache_key] = self.valid_files
                logger.info(f"Cache built with {len(self.valid_files)} valid files")

        # Initialize with precomputed valid files
        super().__init__(
            root=root,
            transform=transform,
            loader=self._safe_loader,
            is_valid_file=lambda path: path in self.valid_files
        )

        # Update samples to only include valid files
        self.samples = [s for s in self.samples if s[0] in self.valid_files]
        self.imgs = self.samples
        logger.info(f"Dataset initialized with {len(self.samples)} samples")

    def _generate_cache_key(self, root):
        """Generate a unique key for the dataset based on its path and structure"""
        # Use dataset path and modification time of a few files to create a key
        try:
            hash_obj = hashlib.md5()
            hash_obj.update(root.encode())

            # Sample a few files to check for changes
            sample_files = []
            for root_dir, _, files in os.walk(root):
                if len(sample_files) >= 100:  # Sample up to 100 files
                    break
                for file in files[:10]:  # Sample first 10 files from each directory
                    if len(sample_files) >= 100:
                        break
                    sample_files.append(os.path.join(root_dir, file))

            for file in sorted(sample_files):
                if os.path.exists(file):
                    hash_obj.update(file.encode())
                    hash_obj.update(str(os.path.getmtime(file)).encode())

            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}. Using fallback.")
            return hashlib.md5(root.encode()).hexdigest()

    def _build_cache(self):
        """Build cache of valid files efficiently"""
        valid_files = set()
        total_files = 0

        # Count total files first for progress reporting
        for root_dir, _, files in os.walk(self.root):
            total_files += len(files)

        logger.info(f"Scanning {total_files} files in {self.root}")
        processed_files = 0
        start_time = time.time()

        for root_dir, _, files in os.walk(self.root):
            for file in files:
                processed_files += 1
                if processed_files % 1000 == 0:
                    elapsed = time.time() - start_time
                    files_per_sec = processed_files / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_files}/{total_files} files "
                                f"({files_per_sec:.1f} files/sec)")

                file_path = os.path.join(root_dir, file)
                if self._is_valid_file_fast(file_path):
                    valid_files.add(file_path)

        return valid_files

    def _is_valid_file_fast(self, path):
        """Fast file validation without full image decoding"""
        try:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) == 0:
                logger.debug(f"Empty file: {path}")
                return False

            # Quick check for common image file signatures
            with open(path, 'rb') as f:
                header = f.read(12)
                # Check for JPEG, PNG, and other common formats
                if header.startswith(b'\xff\xd8\xff'):  # JPEG
                    return True
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                    return True
                elif header.startswith(b'MM\x00\x2a') or header.startswith(b'II\x2a\x00'):  # TIFF
                    return True
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):  # GIF
                    return True
                elif header.startswith(b'\x00\x00\x00\x0cJXL \r\n\x87\n'):  # JPEG XL
                    return True
                elif header[0:4] == b'RIFF' and header[8:12] == b'WEBP':
                    # WebP
                    return True
                else:
                    # Fallback to full verification for uncommon formats
                    Image.open(path).verify()
                    return True

        except Exception as e:
            logger.debug(f"Invalid file {path}: {str(e)}")
            return False



    def _load_cache(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Verify cache is not stale
                    if cache_data.get('cache_key') == self.cache_key:
                        return set(cache_data['valid_files'])
            except (EOFError, pickle.UnpicklingError) as e:
                logger.warning(f"Corrupted cache file {self.cache_file}: {e}")
                os.remove(self.cache_file)  # Remove corrupted cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None


    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_data = {
                'cache_key': self.cache_key,
                'valid_files': list(self.valid_files),
                'timestamp': time.time()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


    def _safe_loader(self, path):
        """Safe image loader with fallback"""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()
                return img.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            # Return placeholder image
            return Image.new('RGB', (256, 256), (0, 0, 0))


# Keep the rest of your classes and functions but update to use FastImageFolder
def setup_dataloader(
        folder_path,
        batch_size=20,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        is_train=True,
        is_distributed=None,
        cache_dir=".dataset_cache",
        skip_validation=False
):
    """
    Create data loader with optimized caching
    """
    # Auto-detect distributed mode
    if is_distributed is None:
        is_distributed = dist.is_initialized() if dist.is_available() else False

    # Use caching for validation if we're in distributed mode
    use_cache_for_validation = is_distributed and not is_train

    dataset = FastImageFolder(
        folder_path,
        transform=get_transforms(is_train),
        cache_dir=cache_dir,
        skip_validation=skip_validation or use_cache_for_validation
    )

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset)
        logger.info(f"Initialized distributed sampler (rank={dist.get_rank()}, world_size={dist.get_world_size()})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if is_train else 2,
        collate_fn=lambda x: torch.utils.data.default_collate(
            [item for item in x if item[0] is not None]  # Filter invalid data
        )
    )
    return loader


# Add a function to pre-cache datasets
def pre_cache_dataset(folder_path, cache_dir=".dataset_cache"):
    """Pre-cache a dataset to avoid validation during training"""
    logger.info(f"Pre-caching dataset: {folder_path}")
    dataset = FastImageFolder(folder_path, cache_dir=cache_dir, skip_validation=False)
    logger.info(f"Pre-caching complete. Found {len(dataset.samples)} valid files.")
    return dataset

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

# def setup_dataloader(
#     folder_path,
#     batch_size=20,
#     num_workers=16,
#     pin_memory=True,
#     drop_last=True,
#     is_train=True,
#     is_distributed=None
# ):
#     """
#     创建数据加载器
#     Args:
#         is_distributed: 强制指定是否分布式模式(None时自动检测)
#     """
#     # 自动检测分布式模式
#     if is_distributed is None:
#         is_distributed = dist.is_initialized() if dist.is_available() else False
#
#     dataset = RobustImageFolder(
#         folder_path,
#         transform=get_transforms(is_train)
#     )
#
#     sampler = None
#     if is_distributed:
#         sampler = DistributedSampler(dataset)
#         logger.info(f"初始化分布式采样器 (rank={dist.get_rank()}, world_size={dist.get_world_size()})")
#
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(sampler is None and is_train),
#         sampler=sampler,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#         persistent_workers=num_workers > 0,
#         collate_fn=lambda x: torch.utils.data.default_collate(
#             [item for item in x if item[0] is not None]  # 过滤无效数据
#         )
#     )
#     return loader

def setup_infinity_train_dataloader(
    folder_path,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    is_distributed=None,
    skip_validation = False,
    cache_dir = '.dataset_cache'
):
    """无限数据流加载器"""
    loader = setup_dataloader(
        folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        is_train=True,
        is_distributed=is_distributed,
        skip_validation = skip_validation,
        cache_dir = cache_dir
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
    is_distributed=None,
    skip_validation = False,
    cache_dir = '.dataset_cache',
):
    """验证集加载器"""
    return setup_dataloader(
        folder_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        is_train=False,
        # is_distributed=is_distributed
        is_distributed = False, # avoid distributed processing when running evaluation
        skip_validation = skip_validation,
        cache_dir = cache_dir
    )
