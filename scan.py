import os
from tqdm import tqdm

def check_directory(directory):
    """递归检查目录下的所有文件"""
    empty_files = []
    corrupt_images = []
    other_files = []
    
    # 遍历所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files, desc=f"检查 {root}"):
            file_path = os.path.join(root, file)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"⚠️ 文件不存在: {file_path}")
                continue
                
            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                empty_files.append(file_path)
                continue
                
            # 如果是图片文件，检查是否损坏
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    from PIL import Image
                    with open(file_path, 'rb') as f:
                        img = Image.open(f)
                        img.verify()  # 验证图片完整性
                except Exception as e:
                    corrupt_images.append((file_path, str(e)))
            else:
                other_files.append(file_path)
    
    # 打印报告
    print("\n" + "="*50)
    print(f"检查完成: {directory}")
    print(f"总文件数: {len(empty_files)+len(corrupt_images)+len(other_files)}")
    print(f"空文件数: {len(empty_files)}")
    print(f"损坏图片数: {len(corrupt_images)}")
    print(f"其他文件数: {len(other_files)}")
    
    if empty_files:
        print("\n空文件列表(最多显示10个):")
        for f in empty_files[:10]:
            print(f"  {f}")
    
    if corrupt_images:
        print("\n损坏图片列表(最多显示10个):")
        for f, e in corrupt_images[:10]:
            print(f"  {f}: {e}")

if __name__ == "__main__":
    target_dir = "data/GenImage"  # 替换为你要检查的目录
    check_directory(target_dir)