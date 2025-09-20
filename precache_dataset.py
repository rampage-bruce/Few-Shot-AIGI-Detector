from datasets import pre_cache_dataset
import os

data_root = 'data/GenImage'
folders = ['real', 'ADM', 'BigGAN', 'glide', 'Midjourney', 'SD', 'VQDM']
cache_dir = '.dataset_cache_2'

for folder in folders:
    train_path = os.path.join(data_root, folder, 'train')
    val_path = os.path.join(data_root, folder, 'val')

    if os.path.exists(train_path):
        print(f'Caching {train_path}')
        pre_cache_dataset(train_path, cache_dir=cache_dir)

    if os.path.exists(val_path):
        print(f'Caching {val_path}')
        pre_cache_dataset(val_path, cache_dir=cache_dir)