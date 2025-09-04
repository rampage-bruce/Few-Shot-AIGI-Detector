import os
import shutil
from tqdm import tqdm


def move_files_with_progress(source_root, target_root):
    """移动文件并显示进度条"""
    # 扩展~为绝对路径
    source_root = os.path.expanduser(source_root)
    target_root = os.path.expanduser(target_root)

    # 创建目标目录（如果不存在）
    os.makedirs(target_root, exist_ok=True)

    print(f'\nCopying: {source_root} -> {target_root}')
    files = os.listdir(source_root)

    for filename in tqdm(files, desc="Moving files"):
        try:
            src_path = os.path.join(source_root, filename)
            dst_path = os.path.join(target_root, filename)
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"\nError moving {filename}: {str(e)}")


def copy_folders_with_progress(source_root, target_root):
    """复制文件夹并显示进度条"""
    try:
        # 扩展~为绝对路径
        source_root = os.path.expanduser(source_root)
        target_root = os.path.expanduser(target_root)

        # 检查源目录是否存在
        if not os.path.exists(source_root):
            print(f"\nWarning: Source directory does not exist - {source_root}")
            return

        # 创建目标目录（如果不存在）
        os.makedirs(target_root, exist_ok=True)

        print(f'\nCopying folders from: {source_root} -> {target_root}')

        # 获取所有子文件夹（排除文件）
        folders = [f for f in os.listdir(source_root)
                   if os.path.isdir(os.path.join(source_root, f))]

        for folder in tqdm(folders, desc="Copying folders"):
            try:
                src_path = os.path.join(source_root, folder)
                dst_path = os.path.join(target_root, folder)

                # 使用copytree复制整个文件夹
                shutil.copytree(src_path, dst_path)
            except shutil.Error as e:
                print(f"\nError copying {folder}: {str(e)}")
            except Exception as e:
                print(f"\nUnexpected error with {folder}: {str(e)}")
    except Exception as e:
        print(f"\nFatal error in copy_folders_with_progress: {str(e)}")


# ADM (need to rerun this dataset)
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/ADM/imagenet_ai_0508_adm/train/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/ADM/train/ai')
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/ADM/imagenet_ai_0508_adm/val/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/ADM/val/ai')

# BigGAN
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/BigGAN/imagenet_ai_0419_biggan/train/ai',
#                       '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/BigGAN/train/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/BigGAN/imagenet_ai_0419_biggan/val/ai',
#                       '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/BigGAN/val/ai')

# Midjourney
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/Midjourney/imagenet_midjourney/train/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/Midjourney/train/ai')
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/Midjourney/imagenet_midjourney/val/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/Midjourney/val/ai')

# VQDM
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/VQDM/imagenet_ai_0419_vqdm/train/ai',
#                       '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/VQDM/train/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/VQDM/imagenet_ai_0419_vqdm/val/ai',
#                       '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/VQDM/val/ai')

# glide
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/glide/imagenet_glide/train/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/glide/train/ai')
move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/glide/imagenet_glide/val/ai',
                      '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/glide/val/ai')


# not executed
# move_files_with_progress('~/autodl-tmp/GenImage/wukong/imagenet_ai_0424_wukong/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train/nature')
# move_files_with_progress('~/autodl-tmp/GenImage/wukong/imagenet_ai_0424_wukong/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val/nature')

# real
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/real/train/nature')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/nature',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/real/val/nature')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/nature',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/real/train/nature')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val/nature',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/real/val/nature')

# SD
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/train/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/val/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/train/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/val/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/wukong/imagenet_ai_0424_wukong/train/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/train/ai')
# move_files_with_progress('/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/wukong/imagenet_ai_0424_wukong/val/ai',
#                          '/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage/SD/val/ai')

# real
# move_files_with_progress('~/autodl-tmp/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train')
# move_files_with_progress('~/autodl-tmp/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val')
# move_files_with_progress('~/autodl-tmp/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train')
# move_files_with_progress('~/autodl-tmp/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val')

# move_files_with_progress('~/autodl-tmp/GenImage/wukong/imagenet_ai_0424_wukong/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train/SD')
# move_files_with_progress('~/autodl-tmp/GenImage/wukong/imagenet_ai_0424_wukong/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val/SD')

# move_files_with_progress('~/autodl-tmp/GenImage/glide/imagenet_glide/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train/glide')
# move_files_with_progress('~/autodl-tmp/GenImage/glide/imagenet_glide/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val/glide')

# move_files_with_progress('~/autodl-tmp/GenImage/VQDM/imagenet_ai_0419_vqdm/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train/VQDM')
# move_files_with_progress('~/autodl-tmp/GenImage/VQDM/imagenet_ai_0419_vqdm/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val/VQDM')

# move_files_with_progress('~/autodl-tmp/GenImage/Midjourney/imagenet_midjourney/train/nature',
#                       '~/autodl-tmp/GenImage2/real/train/Midjourney')
# move_files_with_progress('~/autodl-tmp/GenImage/Midjourney/imagenet_midjourney/val/nature',
#                       '~/autodl-tmp/GenImage2/real/val/Midjourney')


print("\nAll file moving operations completed!")