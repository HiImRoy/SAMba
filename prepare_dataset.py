
import os
from PIL import Image
import glob

def convert_dataset(dataset_path):
    """
    转换指定数据集的图像和掩码文件。

    1. 将 'img' 目录下的 .JPG 文件后缀统一为 .jpg。
    2. 将 'lab' 目录下的 .bmp 掩码文件转换为单通道灰度 .png 文件。

    :param dataset_path: 数据集根目录的路径 (例如, 'data/CrackTree260')。
    """
    print(f"开始处理数据集: {dataset_path}")

    # --- 1. 标准化图像文件后缀 ---
    img_dir = os.path.join(dataset_path, 'img')
    if os.path.isdir(img_dir):
        print(f"\n正在检查 '{img_dir}' 中的图像文件后缀...")
        # 使用 glob 匹配不区分大小写的 .JPG
        jpg_files_upper = glob.glob(os.path.join(img_dir, '*.[jJ][pP][gG]'))
        
        renamed_count = 0
        for filepath in jpg_files_upper:
            if filepath.endswith('.JPG'):
                new_filepath = os.path.splitext(filepath)[0] + '.jpg'
                os.rename(filepath, new_filepath)
                print(f"  重命名: {os.path.basename(filepath)} -> {os.path.basename(new_filepath)}")
                renamed_count += 1
        if renamed_count == 0:
            print("  所有图像文件后缀已是 .jpg，无需更改。")
        else:
            print(f"  完成，共重命名 {renamed_count} 个文件。")
    else:
        print(f"  警告: 图像目录 '{img_dir}' 不存在，跳过处理。")


    # --- 2. 转换掩码文件格式 ---
    lab_dir = os.path.join(dataset_path, 'lab')
    if os.path.isdir(lab_dir):
        print(f"\n正在转换 '{lab_dir}' 中的掩码文件 (BMP -> PNG)...")
        bmp_files = glob.glob(os.path.join(lab_dir, '*.bmp'))
        
        if not bmp_files:
            print("  未找到 .bmp 文件，无需转换。")
        else:
            converted_count = 0
            for bmp_path in bmp_files:
                try:
                    # 构建新的png文件名
                    png_path = os.path.splitext(bmp_path)[0] + '.png'

                    # 打开BMP图像并转换为灰度图 ('L' mode)
                    with Image.open(bmp_path) as img:
                        grayscale_img = img.convert('L')
                        grayscale_img.save(png_path, 'PNG')
                    
                    # 删除原始的BMP文件
                    os.remove(bmp_path)
                    
                    print(f"  转换: {os.path.basename(bmp_path)} -> {os.path.basename(png_path)}")
                    converted_count += 1
                except Exception as e:
                    print(f"  处理文件 {os.path.basename(bmp_path)} 时出错: {e}")
            print(f"  完成，共转换 {converted_count} 个掩码文件。")
    else:
        print(f"  警告: 标签目录 '{lab_dir}' 不存在，跳过处理。")

    print("\n数据预处理完成！")

if __name__ == '__main__':
    # 设置你的数据集路径
    CRACKTREE_PATH = 'data/CrackTree260'
    
    if not os.path.isdir(CRACKTREE_PATH):
        print(f"错误: 数据集路径 '{CRACKTREE_PATH}' 不存在。")
        print("请确保脚本与 'data' 文件夹在同一目录下，或者修改 'CRACKTREE_PATH' 变量。")
    else:
        convert_dataset(CRACKTREE_PATH)
