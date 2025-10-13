# Copyright (c) Roy. All rights reserved.

import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# --- 任务一：构建包含丰富数据增强的训练流程 ---

class crack_tree_dataset(Dataset):
    """
    用于路面裂缝分割任务的 PyTorch 数据集类。

    该类集成了 albumentations 库，以实现对图像和掩码的高效同步数据增强。
    它适用于输入图像为800x600，模型输入为448x448的场景。

    作者: Roy
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        初始化数据集。

        Args:
            image_paths (list[str]): 图像文件的路径列表。
            mask_paths (list[str]): 对应掩码文件的路径列表。
            transform (A.Compose, optional): 一个 albumentations 的变换管道。默认为 None。
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        assert len(self.image_paths) == len(self.mask_paths), \
            "图像和掩码的数量必须一致"

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        加载、增强并返回一个样本（图像和掩码）。

        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: 包含 (图像张量, 掩码张量) 的元组。
        """
        # a. 加载图像和掩码
        # 使用 OpenCV 加载图像，注意 OpenCV 默认读取格式为 BGR
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换为 RGB

        # 加载单通道的掩码
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # b. 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # c. 转换为 PyTorch 张量 (这一步通常由 ToTensorV2 在 transform 中完成)
        # 图像张量应为 torch.float32 类型并已归一化
        # 掩码张量应为 torch.long 类型，并且不需要通道维度
        # ToTensorV2 会自动处理 image 的 HWC -> CHW 转换和归一化
        # ToTensorV2 会自动处理 mask 的 HW -> HW (或 HWC -> CHW，取决于输入)
        # 我们需要确保 mask 最后是 (H, W) 形状的 LongTensor
        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            mask = torch.squeeze(mask, 0) # 从 (1, H, W) 降维到 (H, W)

        return image, mask.long()


# --- 任务二：实现带权重融合的滑动窗口推理流程 ---

def _generate_gaussian_weights(patch_size, sigma_ratio=1./8.):
    """
    生成一个二维高斯权重图。
    中心权重最高，向边缘逐渐衰减，用于平滑拼接。
    """
    center = patch_size / 2
    sigma = patch_size * sigma_ratio
    x = np.arange(0, patch_size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = center
    # 计算二维高斯分布
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

def predict_with_weighted_sliding_window(model, image_path, patch_size=448, stride=224, device='cuda'):
    """
    在完整的高分辨率图像上，使用带高斯权重融合的滑动窗口进行平滑推理。

    Args:
        model (nn.Module): 训练好的 PyTorch 模型。
        image_path (str): 待预测的完整图像路径。
        patch_size (int): 滑动窗口的尺寸，应与模型输入尺寸一致。
        stride (int): 滑动窗口的步长。步长小于窗口尺寸以保证重叠。
        device (str): 推理设备, 'cuda' 或 'cpu'。

    Returns:
        np.ndarray: 返回一个与原图尺寸相同的、平滑的分割概率图 (float32)。
    """
    # 1. 加载图像并进行基础预处理
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w, _ = image.shape

    # 使用 albumentations 进行归一化和张量转换
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # 2. 生成二维高斯权重图
    gaussian_weights = _generate_gaussian_weights(patch_size)
    gaussian_weights = torch.from_numpy(gaussian_weights).to(device).float()

    # 3. 创建累加器
    prediction_accumulator = torch.zeros((1, original_h, original_w), device=device, dtype=torch.float32)
    weight_accumulator = torch.zeros((original_h, original_w), device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # 4. 滑动窗口遍历
        for y in range(0, original_h, stride):
            for x in range(0, original_w, stride):
                # a. 确定图像块的边界，并处理边缘情况
                y_end = min(y + patch_size, original_h)
                x_end = min(x + patch_size, original_w)
                y_start = max(0, y_end - patch_size)
                x_start = max(0, x_end - patch_size)

                patch = image_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # b. 模型推理
                # 假设模型输出为 (B, 1, H, W) 的 logits
                patch_pred = model(patch)
                patch_pred = torch.sigmoid(patch_pred) # 转换为概率
                patch_pred = patch_pred.squeeze(0) # (1, H, W)

                # c. 累加加权预测和权重
                prediction_accumulator[:, y_start:y_end, x_start:x_end] += patch_pred * gaussian_weights
                weight_accumulator[y_start:y_end, x_start:x_end] += gaussian_weights

    # 5. 归一化得到最终结果
    # 处理除零问题，在权重为0的区域保持预测值为0
    final_prediction = prediction_accumulator / (weight_accumulator + 1e-8)
    
    # 6. 返回Numpy数组
    return final_prediction.squeeze(0).cpu().numpy()


# --- 示例代码 ---
if __name__ == '__main__':
    # 假设我们有一个数据集目录结构如下:
    # /data/crack_dataset/
    #   ├── images/
    #   │   ├── 001.jpg
    #   │   └── 002.jpg
    #   └── masks/
    #       ├── 001.bmp
    #       └── 002.bmp

    # 1. 定义用于训练的 albumentations 变换管道
    print("--- 任务一：演示数据增强和数据集创建 ---")
    train_transform = A.Compose([
        A.RandomCrop(height=448, width=448, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 2. 准备文件路径 (这里用占位符代替)
    # 在实际使用中，您需要用 glob 或 os.walk 来获取真实的文件列表
    data_root = "/data/crack_dataset"
    image_files = [os.path.join(data_root, "images", "001.jpg")]
    mask_files = [os.path.join(data_root, "masks", "001.bmp")]

    # 3. 实例化数据集
    train_dataset = crack_tree_dataset(
        image_paths=image_files,
        mask_paths=mask_files,
        transform=train_transform
    )

    print(f"数据集大小: {len(train_dataset)}")

    # 4. 获取一个样本并检查其形状和类型
    if len(train_dataset) > 0:
        img_tensor, mask_tensor = train_dataset[0]
        print(f"增强后的图像张量形状: {img_tensor.shape}, 类型: {img_tensor.dtype}")
        print(f"增强后的掩码张量形状: {mask_tensor.shape}, 类型: {mask_tensor.dtype}")
        # 预期输出:
        # 增强后的图像张量形状: torch.Size([3, 448, 448]), 类型: torch.float32
        # 增强后的掩码张量形状: torch.Size([448, 448]), 类型: torch.int64

    print("\n--- 任务二：演示滑动窗口推理函数调用 ---")

    # 5. 创建一个虚拟模型用于演示
    # 这里的模型直接输出一个和输入相同尺寸的随机张量
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn_like(x)[:, 0:1, :, :]

    dummy_model = DummyModel().to('cuda' if torch.cuda.is_available() else 'cpu')

    # 6. 创建一个虚拟的800x600图像文件
    dummy_image_path = "dummy_800x600.jpg"
    cv2.imwrite(dummy_image_path, np.zeros((600, 800, 3), dtype=np.uint8))

    # 7. 调用滑动窗口推理函数
    print(f"正在对图像 '{dummy_image_path}' 进行滑动窗口推理...")
    smooth_prediction = predict_with_weighted_sliding_window(
        model=dummy_model,
        image_path=dummy_image_path,
        patch_size=448,
        stride=224, # 50% 重叠率
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"推理完成，生成的平滑预测图形状: {smooth_prediction.shape}")
    # 预期输出:
    # 推理完成，生成的平滑预测图形状: (600, 800)

    # 8. 清理虚拟文件
    os.remove(dummy_image_path)
