import os
import hashlib
from collections import defaultdict
from tqdm import tqdm

data_root = "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage"  # Replace with your data root path
hash_dict = defaultdict(list)

def hash_file(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing file {file_path}: {e}")
        return None

print("checking image leakage across all datasets...")
for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if os.path.isdir(class_path):
        for split in ['train', 'val']:
            split_path = os.path.join(class_path, split)
            if os.path.exists(split_path):
                print(split_path)
                for file in os.listdir(split_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(split_path, file)
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        hash_dict[file_hash].append((file_path, class_name, split))

# Find duplicates
for hash, files in hash_dict.items():
    if len(files) > 1:
        print(f"Duplicate hash {hash} found in:")
        for file in files:
            print(f"  {file}")

print("checking image leakage between train and val split...")

for class_name in os.listdir(data_root):
    if class_name == "real":
        subfix = "nature"
    else:
        subfix = "ai"
    train_path = os.path.join(data_root, class_name, 'train',subfix)
    val_path = os.path.join(data_root, class_name, 'val',subfix)
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(train_path)
        print(val_path)
        train_files = set(os.listdir(train_path))
        val_files = set(os.listdir(val_path))
        common = train_files.intersection(val_files)
        if common:
            print(f"Class {class_name} has overlapping files between train and val: {common}")


print("checking real data directory...")

def check_real_data_sources(genimage_root, sd_v14_path, sd_v15_path):
    """
    Verify that real data only contains nature images from Stable Diffusion v1.4 and v1.5

    Args:
        genimage_root: Path to the GenImage dataset root
        sd_v14_path: Path to Stable Diffusion v1.4 dataset
        sd_v15_path: Path to Stable Diffusion v1.5 dataset
    """
    # Get all nature images from SD v1.4 and v1.5
    sd_nature_images = set()

    # Collect SD v1.4 nature images
    if os.path.exists(sd_v14_path):
        nature_path_v14 = os.path.join(sd_v14_path, "train", "nature")
        if os.path.exists(nature_path_v14):
            print("Adding sd14...")
            for img in tqdm(os.listdir(nature_path_v14)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sd_nature_images.add(hash_file(os.path.join(nature_path_v14, img)))

    # Collect SD v1.5 nature images
    if os.path.exists(sd_v15_path):
        nature_path_v15 = os.path.join(sd_v15_path, "train", "nature")
        if os.path.exists(nature_path_v15):
            print("Adding sd15...")
            for img in tqdm(os.listdir(nature_path_v15)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sd_nature_images.add(hash_file(os.path.join(nature_path_v15, img)))

    # Check all real images in GenImage
    real_path = os.path.join(genimage_root, "real")
    non_nature_images = []

    for split in ["train", "val"]:
        split_path = os.path.join(real_path, split)
        if os.path.exists(split_path):
            print("verifying real directory...")
            for img in tqdm(os.listdir(split_path)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_hash = hash_file(os.path.join(split_path, img))
                    if img_hash not in sd_nature_images:
                        non_nature_images.append(os.path.join(split, img))

    # Report results
    if non_nature_images:
        print(f"Found {len(non_nature_images)} images in 'real' that are not from SD nature folders:")
        for img in non_nature_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(non_nature_images) > 10:
            print(f"  ... and {len(non_nature_images) - 10} more")
    else:
        print("All images in 'real' are from SD nature folders")


# def hash_file(file_path):
#     """Calculate MD5 hash of a file"""
#     hash_md5 = hashlib.md5()
#     with open(file_path, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()


# Usage
genimage_root = "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/GenImage"
sd_v14_path = "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"
sd_v15_path = "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"

check_real_data_sources(genimage_root, sd_v14_path, sd_v15_path)


print("checking individual dataset...")


def verify_dataset_integrity(genimage_root, source_mapping):
    """
    Verify that each dataset's train/ai and val/ai directories only contain data from their designated source

    Args:
        genimage_root: Path to the GenImage dataset root
        source_mapping: Dictionary mapping dataset names to their source paths
    """
    # Create a mapping of source images to their datasets
    source_hashes = defaultdict(set)

    # Collect all source images from each dataset's ai directories
    for dataset_name, source_paths in source_mapping.items():
        if not isinstance(source_paths, list):
            source_paths = [source_paths]

        print("Adding individual sources...")
        for source_path in tqdm(source_paths):
            # Check both train/ai and val/ai in the source
            for split in ["train", "val"]:
                source_ai_path = os.path.join(source_path, split, "ai")
                if os.path.exists(source_ai_path):
                    for img in os.listdir(source_ai_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(source_ai_path, img)
                            img_hash = hash_file(file_path)
                            if img_hash:
                                source_hashes[dataset_name].add(img_hash)

    # Check each dataset in GenImage
    print("Checking datasets...")
    for dataset_name in tqdm(source_mapping.keys()):
        dataset_path = os.path.join(genimage_root, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} not found in {genimage_root}")
            continue

        foreign_images = []

        # Check both train/ai and val/ai in the GenImage dataset
        for split in ["train", "val"]:
            ai_path = os.path.join(dataset_path, split, "ai")
            if not os.path.exists(ai_path):
                print(f"Path {ai_path} not found")
                continue

            for img in os.listdir(ai_path):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(ai_path, img)
                    img_hash = hash_file(img_path)

                    if not img_hash:
                        continue

                    # Check if this image belongs to this dataset
                    if img_hash not in source_hashes[dataset_name]:
                        foreign_images.append(os.path.join(split, "ai", img))

        # Report results for this dataset
        if foreign_images:
            print(f"Found {len(foreign_images)} foreign images in '{dataset_name}':")
            for img in foreign_images[:10]:  # Show first 10
                print(f"  - {img}")
            if len(foreign_images) > 10:
                print(f"  ... and {len(foreign_images) - 10} more")
        else:
            print(f"All images in '{dataset_name}' are from its designated source")


# Define the source mapping for each dataset
source_mapping = {
    "SD": [
        "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4",
        "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5",
        "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/wukong/imagenet_ai_0424_wukong"
    ],
    "ADM": "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/ADM/imagenet_ai_0508_adm",
    "BigGAN": "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/BigGAN/imagenet_ai_0419_biggan",
    "glide": "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/glide/imagenet_glide",
    "Midjourney": "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/Midjourney/imagenet_midjourney",
    "VQDM": "/share/workspace2/haosheng/Few-Shot-AIGI-Detector/data/VQDM/imagenet_ai_0419_vqdm"
    # Add other datasets as needed
}

# Usage
verify_dataset_integrity(genimage_root, source_mapping)