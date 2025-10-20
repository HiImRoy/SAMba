import torch.nn as nn

class DSC(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Convolution)
    由一个深度卷积和一个逐点卷积组成。
    """
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        """
        初始化函数
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param k_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充
        """
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        # 深度卷积：对每个输入通道独立进行卷积，groups=c_in
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        # 逐点卷积：1x1卷积，用于组合深度卷积的输出，改变通道数
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        # 假设输入 x 的维度为 (B, c_in, H, W)
        out = self.dw(x)  # -> (B, c_in, H', W')，H'和W'由步长和填充决定
        out = self.pw(out)  # -> (B, c_out, H', W')
        return out

# pw dw
class IDSC(nn.Module):
    """
    倒置深度可分离卷积 (Inverted Depthwise Separable Convolution)
    先进行逐点卷积（通常是升维），再进行深度卷积。
    """
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        """
        初始化函数
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param k_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充
        """
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        # 逐点卷积：1x1卷积，先于深度卷积执行
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)
        # 深度卷积：对每个输入通道独立进行卷积，groups=c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)


    def forward(self, x):
        # 假设输入 x 的维度为 (B, c_in, H, W)
        out = self.pw(x)  # -> (B, c_out, H, W)
        out = self.dw(out)  # -> (B, c_out, H', W')，H'和W'由步长和填充决定
        return out
