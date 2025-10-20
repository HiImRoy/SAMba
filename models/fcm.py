import torch
import torch.nn as nn

from .dsc import DSC, IDSC

class FCM(nn.Module):
    """
    基于特征融合的内容模块 (Feature-fusion-based Content Module)
    """
    def __init__(self, dim):
        """
        初始化函数
        :param dim: 输入和输出特征图的通道数
        """
        super().__init__()
        # 自适应平均池化，用于全局信息提取
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 线性层，用于计算通道注意力权重
        self.linear = nn.Linear(dim, dim)
        # 倒置深度可分离卷积，用于残差连接中的降维
        self.down = IDSC(3*dim, dim)

        # 主要的融合和降维模块
        self.fuse = nn.Sequential(IDSC(3*dim, dim),      # 倒置深度可分离卷积
                                  nn.BatchNorm2d(dim), # 归一化
                                  nn.GELU(),           # 激活函数
                                  DSC(dim, dim),         # 深度可分离卷积
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  DSC(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU()
                                  )

    def forward(self, x1, y1):
        # 假设 x1 和 y1 的维度均为 (B, dim, H, W)
        B1, C1, H1, W1 = x1.shape # C1 = dim
        B2, C2, H2, W2 = y1.shape # C2 = dim

        # --- 通道注意力机制 ---
        x_temp = self.avg(x1)   # -> (B1, C1, 1, 1)
        y_temp = self.avg(y1)   # -> (B2, C2, 1, 1)
        # reshape + linear 计算权重
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1)) # -> (B1, 1, 1, C1)
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2)) # -> (B2, 1, 1, C2)

        # --- 特征重标定 ---
        # 转换维度以进行广播乘法
        x_temp = x1.permute(0, 2, 3, 1) # -> (B1, H1, W1, C1)
        y_temp = y1.permute(0, 2, 3, 1) # -> (B2, H2, W2, C2)
        # 将权重应用到特征图
        x1 = x_temp * x_weight # -> (B1, H1, W1, C1)
        y1 = y_temp * y_weight # -> (B2, H2, W2, C2)

        # --- 多模式融合 ---
        # 模式1: 拼接
        out1 = torch.cat([x1, y1], dim=3) # -> (B, H, W, 2*C)
        # 模式2: 逐元素相乘
        out2 = x1 * y1                   # -> (B, H, W, C)
        # 组合两种模式
        fuse = torch.cat([out1, out2], dim=3) # -> (B, H, W, 3*C)
        # 转换回 (B, C, H, W) 格式
        fuse = fuse.permute(0, 3, 1, 2)       # -> (B, 3*C, H, W)

        # --- 降维与残差连接 ---
        out = self.fuse(fuse)           # -> (B, C, H, W)
        out = out + self.down(fuse)     # 残差连接 -> (B, C, H, W)

        return out
