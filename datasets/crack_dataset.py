# Author: Roy
import os.path
import cv2
import random
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # 引入 functional 以便对图像和标签进行相同的几何变换
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    """
    一个用于裂缝数据集的类，现在包含了动态的80/20分割和针对训练集的数据增强。
    """

    def __init__(self, args):
        """初始化此数据集类。

        这个类现在处理数据集到训练集和验证集的动态分割，基于一个固定的随机种子，
        并为训练阶段定义了数据增强流程。
        """
        BaseDataset.__init__(self, args)
        self.args = args
        self.phase = args.phase

        # --- 1. 定义数据目录 ---
        img_dir = os.path.join(args.dataset_path, 'img')
        self.lab_dir = os.path.join(args.dataset_path, 'lab')

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"图像目录未找到: {img_dir}。请确保您的数据位于 'data/crack500/img/'。")
        if not os.path.isdir(self.lab_dir):
            raise FileNotFoundError(f"标签目录未找到: {self.lab_dir}。请确保您的数据位于 'data/crack500/lab/'。")

        # --- 2. 获取所有图像路径并可复现地打乱 ---
        all_img_paths = sorted(make_dataset(img_dir))
        random.Random(args.seed).shuffle(all_img_paths)

        # --- 3. 计算80%的分割点 ---
        split_idx = int(len(all_img_paths) * 0.8)
        if len(all_img_paths) == 0:
            raise ValueError(f"在 {img_dir} 中未找到任何图像。")

        # --- 4. 根据当前阶段分配路径切片 ---
        if args.phase == 'train':
            self.img_paths = all_img_paths[:split_idx]
            print(f"数据集: 使用 {len(self.img_paths)} 张图像进行训练 (总数的80%)。")
        elif args.phase == 'test':
            self.img_paths = all_img_paths[split_idx:]
            print(f"数据集: 使用 {len(self.img_paths)} 张图像进行验证 (总数的20%)。")
        else:
            self.img_paths = all_img_paths

        # --- 5. 定义基础的Tensor转换 ---
        # 这些转换将在数据增强之后应用
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mask_to_tensor = MaskToTensor()

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。"""
        # --- 1. 加载图像和标签路径 ---
        img_path = self.img_paths[index]
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        # 灵活处理 .jpg 和 .png 标签
        lab_path_jpg = os.path.join(self.lab_dir, base_filename + '.jpg')
        lab_path_png = os.path.join(self.lab_dir, base_filename + '.png')
        lab_path = lab_path_jpg if os.path.exists(lab_path_jpg) else lab_path_png
        if not os.path.exists(lab_path):
            raise FileNotFoundError(f"找不到图像 {img_path} 对应的标签文件。已查找: {lab_path_jpg} 和 {lab_path_png}")

        # --- 2. 使用 OpenCV 读取图像和标签 ---
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        # --- 3. 预处理和尺寸调整 ---
        # 确保图像是3通道BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 确保标签是单通道灰度图
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # 调整尺寸
        w, h = self.args.load_width, self.args.load_height
        if w > 0 and h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST) # 标签使用最近邻插值

        # 将标签二值化为 0 或 255
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)

        # --- 4. 转换为 PIL Image 以便进行数据增强 ---
        # OpenCV 读取的是 BGR, PyTorch 模型需要 RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lab_pil = Image.fromarray(lab)

        # --- 5. 应用数据增强 (仅对训练集) ---
        if self.phase == 'train':
            # a. 随机水平翻转
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                lab_pil = TF.hflip(lab_pil)

            # b. 随机颜色抖动 (已禁用)
            # img_pil = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(img_pil)

            # c. 随机仿射变换 (对图像和标签应用相同的变换)
            affine_params = transforms.RandomAffine.get_params(
                degrees=[-10, 10],
                translate=(0.1, 0.1),
                scale_ranges=(0.9, 1.1),
                shears=None,
                img_size=img_pil.size
            )
            img_pil = TF.affine(img_pil, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
            lab_pil = TF.affine(lab_pil, *affine_params, interpolation=TF.InterpolationMode.NEAREST)

            # d. (可选) 弹性变换
            # 注意: 弹性变换要保证图像和标签采用完全相同的随机位移场，
            # 使用 torchvision 的标准类实现较为复杂。为保证正确性，我们优先加入
            # 上述已实现的增强方法。如果仍有过拟合，可以进一步探索此方法。

        # --- 6. 最终转换为 Tensor ---
        # 图像: 转为 Tensor 并归一化
        img_tensor = self.to_tensor(img_pil)
        img_tensor = self.normalize(img_tensor)

        # 标签: 再次二值化 (确保增强后仍是 0/1), 然后转为 Tensor
        lab_np = np.array(lab_pil)
        _, lab_np = cv2.threshold(lab_np, 127, 1, cv2.THRESH_BINARY)
        lab_tensor = self.mask_to_tensor(lab_np).unsqueeze(0)

        return {'image': img_tensor, 'label': lab_tensor, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.img_paths)
