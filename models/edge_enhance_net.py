import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddle
from paddle.nn import functional as F
import random
from paddle.io import Dataset
from visualdl import LogWriter
from paddle.vision.transforms import transforms as T
import warnings
warnings.filterwarnings("ignore")
import time
from time import *
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import paddle.nn as nn


# patch merging 模块，用于降低特征图的空间分辨率，同时增加通道数
class patchmerging(nn.Layer):
    """
    Patch Merging层。
    将一个[B, C, H, W]的张量转换为[B, C, H/2, W/2]的张量。
    它通过将输入张量在空间维度上分块并连接，然后通过一个线性层来减少通道维度来实现。
    这是一种常见的下采样操作，用于分层视觉Transformer中。
    """
    def __init__(self, ch_in):
        """
        初始化Patch Merging层。
        :param ch_in: 输入特征图的通道数。
        """
        super(patchmerging, self).__init__()
        # 线性层，将4倍的输入通道数降维回原始通道数
        self.reduction=nn.Linear(ch_in*4,ch_in)
        # 层归一化，对合并后的特征进行归一化
        self.norm=nn.LayerNorm(ch_in*4)
    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 [B, C, H, W]。
        :return: 输出张量，形状为 [B, C, H//2, W//2]。
        """
        # 获取输入张量的维度
        B,C,H,W=x.shape
        # 将输入张量在空间维度上分块
        x0=x[:,:,0::2,0::2] # 左上角
        x1=x[:,:,1::2,0::2] # 左下角
        x2=x[:,:,0::2,1::2] # 右上角
        x3=x[:,:,1::2,1::2] # 右下角
        # 沿着通道维度连接分块后的张量
        out=paddle.concat(x=[x0,x1,x2,x3],axis=1) # [B, 4*C, H/2, W/2]
        # 重塑张量以进行线性变换
        out1=paddle.reshape(out,[B,C*4,-1]) # [B, 4*C, (H/2)*(W/2)]
        out1=paddle.transpose(out1,perm=[0,2,1]) # [B, (H/2)*(W/2), 4*C]
        # 应用层归一化和线性降维
        out2=self.norm(out1)
        out2=self.reduction(out2) # [B, (H/2)*(W/2), C]
        # 恢复张量形状
        out2=paddle.transpose(out2,perm=[0,2,1]) # [B, C, (H/2)*(W/2)]
        out2=paddle.reshape(out2,[B,C,H//2,W//2]) # [B, C, H/2, W/2]
        return out2




import math
# 通道注意力（Channel Attention）模块
class CA(nn.Layer):
    """
    通道注意力（CA）模块。
    该模块通过自适应地重新校准通道特征响应来增强模型的表示能力。
    它首先通过全局平均池化来聚合空间信息，然后使用一维卷积来捕获通道间的依赖关系。
    """
    def __init__(self, ch_in, ch_out,b=1,gama=2):
        """
        初始化通道注意力模块。
        :param ch_in: 输入通道数。
        :param ch_out: 输出通道数 (在此实现中未使用)。
        :param b: 用于计算卷积核大小的偏置参数。
        :param gama: 用于计算卷积核大小的缩放参数。
        """
        super(CA, self).__init__()
        self.ch_in=ch_in
        # 根据输入通道数动态计算一维卷积的核大小
        kernel_size=int(abs((math.log(ch_in,2)+b)/gama))
        # 确保核大小为奇数
        if kernel_size % 2:
            kernel_size=kernel_size
        else:
            kernel_size=kernel_size+1
        
        padding=kernel_size // 2

        # 全局自适应平均池化层，将每个通道的空间维度降为1x1
        self.pool=nn.AdaptiveAvgPool2D(output_size=(1,1))

        # 一维卷积层，用于捕获通道间的依赖关系
        self.conv=nn.Conv1D(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding)
        # Sigmoid激活函数，生成注意力权重
        self.active=nn.Sigmoid()
    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 [B, C, H, W]。
        :return: 经过通道注意力加权后的输出张量，形状与输入相同。
        """
        b,c,h,w=x.shape
        # 全局平均池化
        y=self.pool(x) # [B, C, 1, 1]
        # 重塑和转置以适应一维卷积
        y=paddle.reshape(y,[b,c,-1]) # [B, C, 1]
        y=paddle.transpose(y,perm=[0,2,1]) # [B, 1, C]
        # 一维卷积
        y=self.conv(y) # [B, 1, C]
        # 转置和重塑回原始形状
        y=paddle.transpose(y,perm=[0,2,1]) # [B, C, 1]
        y=paddle.reshape(y,[b,c,1,1]) # [B, C, 1, 1]
        # 应用Sigmoid激活函数生成注意力权重
        y=self.active(y)
        # 将注意力权重应用于输入特征图
        y=y*x
        return y

# 边缘特征提取模块1 (Edge Feature Extraction 1)
class Conv1(nn.Layer):
    """
    第一个卷积模块（EFE1），用于提取多尺度特征。
    它包含并行的1x1, 3x1, 1x3, 3x3卷积，并使用通道注意力。
    """
    def __init__(self, ch_in, ch_out):
        """
        :param ch_in: 输入通道数。
        :param ch_out: 输出通道数。
        """
        super(Conv1, self).__init__()

        self.conv11=nn.Sequential(
            nn.Conv2D(ch_in,ch_in//2,kernel_size=1,stride=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv112=nn.Sequential(
            nn.Conv2D(ch_in*3//2,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
        self.conv113=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.conv31=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv13=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv33=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )
        self.CA=CA(ch_in//2, ch_in//2,b=1,gama=2)
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, ch_in, H, W]。
        :return: 输出张量，形状 [B, ch_out, H, W]。
        """
        x1=self.conv113(x) # 残差连接
        x2=self.conv11(x)
        y=self.conv31(x2)
        y=self.CA(y)
        z=self.conv13(x2)
        z=self.CA(z)
        w=self.conv33(x2)
        out=paddle.concat(x=[y,z,w],axis=1)
        out=self.conv112(out)
        out=out+x1 # 添加残差
        return out

# 边缘特征提取模块2 (Edge Feature Extraction 2)
class Conv2(nn.Layer):
    """
    第二个卷积模块（EFE2），结构与Conv1类似，但输入直接作为残差。
    """
    def __init__(self, ch_in, ch_out):
        """
        :param ch_in: 输入通道数。
        :param ch_out: 输出通道数。
        """
        super(Conv2, self).__init__()

        self.conv11=nn.Sequential(
            nn.Conv2D(ch_in,ch_in//2,kernel_size=1,stride=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv112=nn.Sequential(
            nn.Conv2D(3*ch_in//2,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
        self.conv113=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=1,stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.conv31=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv13=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=(1,3),stride=1,padding=(0,1)),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )

        self.conv33=nn.Sequential(
            nn.Conv2D(ch_in//2,ch_in//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm(ch_in//2),
            nn.ReLU()
        )
        self.CA=CA(ch_in//2, ch_in//2,b=1,gama=2)
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, ch_in, H, W]。
        :return: 输出张量，形状 [B, ch_out, H, W]。
        """
        x1=x # 残差连接
        x2=self.conv11(x)
        y=self.conv31(x2)
        y=self.CA(y)
        z=self.conv13(x2)
        z=self.CA(z)
        w=self.conv33(x2)
        out=paddle.concat(x=[y,z,w],axis=1)
        out=self.conv112(out)
        out=out+x1 # 添加残差
        return out


# 编码器块，包含3个卷积层和1个下采样层
class encoder_block2(nn.Layer):
    def __init__(self, channel_in, channel_out):
        """
        :param channel_in: 输入通道数。
        :param channel_out: 输出通道数。
        """
        super(encoder_block2, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, channel_in, H, W]。
        :return: 输出张量，形状 [B, channel_out, H/2, W/2]。
        """
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.pool(y)
        return y

# 编码器块，包含4个卷积层和1个下采样层
class encoder_block3(nn.Layer):
    def __init__(self, channel_in, channel_out):
        """
        :param channel_in: 输入通道数。
        :param channel_out: 输出通道数。
        """
        super(encoder_block3, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, channel_in, H, W]。
        :return: 输出张量，形状 [B, channel_out, H/2, W/2]。
        """
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.pool(y)
        return y

# 编码器块，包含5个卷积层和1个下采样层
class encoder_block4(nn.Layer):
    def __init__(self, channel_in, channel_out):
        """
        :param channel_in: 输入通道数。
        :param channel_out: 输出通道数。
        """
        super(encoder_block4, self).__init__()
        self.block1 = Conv1(ch_in=channel_in,ch_out=channel_out)
        self.block2 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block3 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block4 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.block5 = Conv2(ch_in=channel_out,ch_out=channel_out)
        self.pool=patchmerging(channel_out)
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, channel_in, H, W]。
        :return: 输出张量，形状 [B, channel_out, H/2, W/2]。
        """
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.pool(y)
        return y


# --- Transformer 相关模块 ---

def conv_1x1_bn(inp, oup):
    """1x1卷积 + BatchNorm + SiLU激活函数"""
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    """nxn卷积 + BatchNorm + SiLU激活函数"""
    return nn.Sequential(
        nn.Conv2D(inp, oup, kernal_size, stride, 1, bias_attr=False),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )

# 前置归一化 (Pre-Normalization)
class PreNorm(nn.Layer):
    """
    在应用函数（如Attention或FeedForward）之前进行层归一化。
    """
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 深度可分离卷积 (Depth-wise Convolution)
class DWC_conv(nn.Layer):
    def __init__(self,c,h,w):
        super().__init__()
        self.h=h
        self.w=w
        self.conv=nn.Conv2D(c,c,kernel_size=3,stride=1,padding=1,groups=c)
    def forward(self,x):
        """
        :param x: 输入张量，形状 [B, P, N, C]。
        :return: 输出张量，形状 [B, P, N, C]。
        """
        B,P,N,C =x.shape
        ts=paddle.transpose(x,perm=[0,3,1,2])
        ts=paddle.reshape(ts,[B,C,-1])
        ts=paddle.reshape(ts,[B,C,self.h,self.w])
        ts = self.conv(ts)
        ts =paddle.reshape(ts,[B,C,-1])
        ts =paddle.transpose(ts,perm=[0,1,2])
        ts=paddle.reshape(ts,[B,P,N,C])
        return ts


# 前馈网络 (Feed Forward Network)
class FeedForward(nn.Layer):
    """
    MLP模块，包含深度可分离卷积。
    """
    def __init__(self, axis, hidden_axis,h,w, dropout=0.):
        super().__init__()
        self.h=h
        self.w=w
        self.fc1 =nn.Linear(axis, hidden_axis)
        self.DW=DWC_conv(hidden_axis,self.h,self.w)
        self.norm=nn.LayerNorm(hidden_axis)
        self.act=nn.GELU()
        self.fc2 = nn.Linear(hidden_axis,axis)
    
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, P, N, axis]。
        :return: 输出张量，形状 [B, P, N, axis]。
        """
        x1 = self.fc1(x)
        x2 = self.DW(x1)
        x2=x1+x2
        x2 = self.norm(x2)
        x2 = self.act(x2)
        x2 = self.fc2(x2)
        return x2

# 注意力机制
class Attention(nn.Layer):
    def __init__(self, axis, heads=8, axis_head=64, dropout=0.):
        super().__init__()
        inner_axis = axis_head *  heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5

        self.attend = nn.Softmax(axis = -1)
        self.to_qkv = nn.Linear(axis, inner_axis * 3, bias_attr = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, P, N, axis]。
        :return: 输出张量，形状 [B, P, N, axis]。
        """
        q,k,v = self.to_qkv(x).chunk(3, axis=-1)

        b,p,n,hd = q.shape
        b,p,n,hd = k.shape
        b,p,n,hd = v.shape
        q = q.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))
        k = k.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))
        v = v.reshape((b, p, n, self.heads, -1)).transpose((0, 1, 3, 2, 4))

        dots = paddle.matmul(q, k.transpose((0, 1, 2, 4, 3))) * self.scale
        attn = self.attend(dots)

        out = (attn.matmul(v)).transpose((0, 1, 3, 2, 4)).reshape((b, p, n,-1))
        return self.to_out(out)

# Transformer 模块
class Transformer(nn.Layer):
    def __init__(self, axis, depth, heads, axis_head, mlp_axis,h,w, dropout=0.):
        super().__init__()
        self.h=h
        self.w=w
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(axis, Attention(axis, heads, axis_head, dropout)),
                PreNorm(axis, FeedForward(axis, mlp_axis,self.h,self.w,dropout))
            ]))
    
    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, P, N, axis]。
        :return: 输出张量，形状 [B, P, N, axis]。
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# MobileViT 块
class MobileViTBlock(nn.Layer):
    def __init__(self, axis, depth, channel, kernel_size, patch_size, mlp_axis,h,w, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.h=h
        self.w=w

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, axis)

        self.transformer = Transformer(axis, depth, 1, 32, mlp_axis,self.h,self.w,dropout)

        self.conv3 = conv_1x1_bn(axis, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)


    def forward(self, x):
        """
        :param x: 输入张量，形状 [B, channel, H, W]。
        :return: 输出张量，形状 [B, channel, H, W]。
        """
        y = x.clone()                      

        # 局部表示
        x = self.conv1(x)
        x = self.conv2(x)
        # 全局表示
        n, c, h, w = x.shape

        x = x.transpose((0,2,3,1)).reshape((n,self.ph * self.pw,-1,c)) # Unfold
        x = self.transformer(x)
        x = x.reshape((n,h,-1,c)).transpose((0,3,1,2)) # Fold

        # 融合
        x = self.conv3(x)
        x = paddle.concat((x, y), 1)
        x = self.conv4(x)
        return x

# 3x3 卷积块
class Convblock1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(Convblock1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        ) 
    def forward(self, x):
        y = self.conv(x)
        return y

# Transformer 编码器块
class transformer(nn.Layer):
    def __init__(self, image_size, axiss,channels, mlp_axis,channel1,h,w, kernel_size=3, patch_size=(2, 2)):
        super(transformer,self).__init__()
        self.h=h
        self.w=w
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = 2        
        self.mv1=MobileViTBlock(axiss, 2 , channels, kernel_size, patch_size,mlp_axis,self.h,self.w)
        self.conv1=Convblock1(channel1, channels)
        self.pool=patchmerging(channels)

    def forward(self, x):
        """
        :param x: 输入张量。
        :return: 经过Transformer和下采样后的输出张量。
        """
        x = self.conv1(x)
        y = self.mv1(x)
        y = self.pool(y)
        return y


# --- 其他功能模块 ---

# 1x1 卷积块
class conv11_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv11_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 3x3 卷积块
class conv33_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv33_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 自适应融合 (Adaptive Fusion) 模块
class AF_block(nn.Layer):
    def __init__(self, ch_in, ch_out,ch_middle):
        super(AF_block, self).__init__()
        self.conv=conv11_block(ch_in*2,ch_in) 
        self.act=nn.Sigmoid()
        self.conv2=conv33_block(ch_in,1)

    def forward(self, a,b):
        """
        自适应地融合两个输入张量 a 和 b。
        :param a: 输入张量1。
        :param b: 输入张量2。
        :return: 融合后的张量。
        """
        c=paddle.concat(x=[a,b],axis=1)
        c=self.conv(c)
        c1 = self.conv2(c)
        c2=self.act(c1)
        out=c2*a+(1-c2)*b
        return out


# 跨层融合 (Cross-Level Fusion) 模块
class CLF_block(nn.Layer):
    def __init__(self, channel_in, channel_out):
        super(CLF_block, self).__init__()
        self.conv1=nn.Conv2D(in_channels=channel_in,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv2=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv3=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.conv4=nn.Conv2D(in_channels=channel_out,out_channels=channel_out,kernel_size=1,stride=1,padding=0)
        self.act=nn.Softmax()
    def forward(self, a,b):
        """
        使用自注意力机制融合两个输入张量 a 和 b。
        :param a: 输入张量1。
        :param b: 输入张量2。
        :return: 融合后的张量。
        """
        z = paddle.concat(x=[a, b], axis=1)
        z = self.conv1(z)
        B,C,H,W=z.shape
        q=self.conv2(z)
        k=self.conv3(z)
        v=self.conv4(z)
        q=paddle.reshape(q,[B,C,-1]) # b c hw
        k=paddle.reshape(k,[B,C,-1]) # b c hw
        v=paddle.reshape(v,[B,C,-1]) # b c hw
        k=paddle.transpose(k,perm=[0,2,1]) # b hw c
        qk=paddle.matmul(q,k) # b c c
        qk=self.act(qk)
        out=paddle.matmul(qk,v) # b c hw
        out=paddle.reshape(out,[B,C,H,W])
        return out


# 多尺度特征提取 (Multi-scale Feature Extraction) 模块
class MFE_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(MFE_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2D(ch_in*2,ch_in,kernel_size=1,stride=1),
            nn.BatchNorm2D(ch_in),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1,dilation=1),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=4,dilation=4),
            nn.BatchNorm2D(ch_out),
            nn.ReLU()
        )
        self.clf1=CLF_block(channel_in=1024,channel_out=512)
        self.clf2=CLF_block(channel_in=1024,channel_out=512)


    def forward(self,a,b):
        """
        融合 a 和 b，并使用不同膨胀率的卷积提取多尺度特征。
        :param a: 输入张量1。
        :param b: 输入张量2。
        :return: 融合并提取多尺度特征后的张量。
        """
        x=paddle.concat(x=[a,b],axis=1)
        x=self.conv(x)
        u=self.conv1(x)
        v=self.conv2(x)
        w=self.conv3(x)
        y = self.clf1(a=u,b=v)
        z = self.clf2(a=y,b=w)
        out=z+x
        return out



# 自适应长程特征提取 (Adaptive Long-range Feature Extraction) 模块
class ALFEblock(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ALFEblock, self).__init__()
        self.conv1=nn.Conv2D(ch_in*2,ch_in,kernel_size=1,stride=1) 
        self.conv2=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv3=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv4=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv5=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.conv6=nn.Conv2D(ch_in,ch_in,kernel_size=1,stride=1)
        self.act=nn.Softmax()
        self.pool1=patchmerging(ch_in)
        self.up=nn.UpsamplingBilinear2D(scale_factor=2)


    def forward(self, x): 
        """
        通过自注意力机制在下采样后的特征图上捕获长程依赖。
        :param x: 输入张量。
        :return: 增强了长程特征的输出张量。
        """
        x1=x
        c1=self.pool1(x)
        
        B,C,H,W=c1.shape
        
        q=self.conv2(c1)
        k=self.conv3(c1)
        v=self.conv4(c1)

        q1=paddle.reshape(q,[B,C,-1]) # B C HW
        k1=paddle.reshape(k,[B,C,-1]) # B C HW
        v=paddle.reshape(v,[B,C,-1])

        q1=paddle.transpose(q1,perm=[0,2,1]) # B HW C
        qk1=paddle.matmul(q1,k1) # B HW HW
        qk1=self.act(qk1)
        qk1=paddle.transpose(qk1,perm=[0,2,1]) # B HW HW
        out1=paddle.matmul(v,qk1)
        out1=paddle.reshape(out1,[B,C,H,W])

        
        q2=self.conv5(c1)
        k2=self.conv6(c1)
        q2=paddle.reshape(q2,[B,C,-1])
        k2=paddle.reshape(k2,[B,C,-1])
        k2=paddle.transpose(k2,perm=[0,2,1])
        qk2=paddle.matmul(q2,k2)
        qk2=self.act(qk2)
        out2=paddle.matmul(qk2,v)
        out2=paddle.reshape(out2,[B,C,H,W])
        
        out=paddle.concat(x=[out1,out2],axis=1)
        out=self.conv1(out)
        out=self.up(out)
        out=out+x1

        return out


# 深度可分离卷积块
class dsconv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(dsconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1,groups=ch_in),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1,groups=ch_out),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 增强残差 (Enhanced Residual) 模块
class ER_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(ER_block, self).__init__()
        self.conv = dsconv_block(ch_in,ch_out)

    def forward(self, x):
        y = self.conv(x)
        y=y+x
        return y



import paddle.nn as nn

# 标准卷积块 (包含两个3x3卷积)
class conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 上采样卷积块
class up_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.up(x)
        return x



import paddle
import numpy as np
import paddle.nn as nn
from PIL import Image
from paddle.autograd import backward
import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.initializer import NumpyArrayInitializer

# 边缘检测卷积
def edge_conv2d(im):
    """
    使用多个Sobel算子进行边缘检测。
    :param im: 输入图像张量，形状 [B, 3, H, W]。
    :return: 边缘图张量，形状 [B, 3, H, W]。
    """
    # Sobel X
    sobel_kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype='float32')
    sobel_kernel=sobel_kernel.reshape((1,1,3,3))
    sobel_kernel=np.repeat(sobel_kernel, 3, axis=1)
    sobel_kernel=np.repeat(sobel_kernel, 3, axis=0)
    conv_op=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel)))
    edge_dect=paddle.pow(conv_op(fluid.dygraph.to_variable(im)),2)

    # Sobel Y
    sobel_kernel1=np.array([[ 1, 2, 1],[ 0, 0, 0],[-1,-2,-1]],dtype='float32')
    sobel_kernel1=sobel_kernel1.reshape((1,1,3,3))
    sobel_kernel1=np.repeat(sobel_kernel1, 3, axis=1)
    sobel_kernel1=np.repeat(sobel_kernel1, 3, axis=0)
    conv_op1=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel1)))
    edge_dect1=paddle.pow(conv_op1(fluid.dygraph.to_variable(im)),2)

    # 45度对角线
    sobel_kernel2=np.array([[ 2, 1, 0],[ 1, 0,-1],[ 0,-1,-2]],dtype='float32')
    sobel_kernel2=sobel_kernel2.reshape((1,1,3,3))
    sobel_kernel2=np.repeat(sobel_kernel2, 3, axis=1)
    sobel_kernel2=np.repeat(sobel_kernel2, 3, axis=0)
    conv_op2=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel2)))
    edge_dect2=paddle.pow(conv_op2(fluid.dygraph.to_variable(im)),2)

    # 135度对角线
    sobel_kernel3=np.array([[ 0,-1,-2],[ 1, 0,-1],[ 2, 1, 0]],dtype='float32')
    sobel_kernel3=sobel_kernel3.reshape((1,1,3,3))
    sobel_kernel3=np.repeat(sobel_kernel3, 3, axis=1)
    sobel_kernel3=np.repeat(sobel_kernel3, 3, axis=0)
    conv_op3=nn.Conv2D(3,3,kernel_size=3,padding=1,weight_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel_kernel3)))
    edge_dect3=paddle.pow(conv_op3(fluid.dygraph.to_variable(im)),2)
    
    sobel_out = edge_dect+edge_dect1+edge_dect2+edge_dect3
    sobel_out=paddle.sqrt(sobel_out)
    return sobel_out



# 主网络模型
class Three_Net(nn.Layer):
    """
    整个网络结构，一个双分支的U-Net变体。
    一个分支是基于卷积的编码器，用于提取边缘特征。
    另一个分支是基于Transformer的编码器，用于提取全局上下文特征。
    两个分支的特征在不同尺度上进行融合。
    """
    def __init__(self, img_ch=3, output_ch=1):
        """
        :param img_ch: 输入图像通道数。
        :param output_ch: 输出分割图通道数。
        """
        super(Three_Net, self).__init__()
        
        # --- 卷积分支 (边缘特征) ---
        self.block11=conv_block(ch_in=3,ch_out=64)
        self.pool=patchmerging(ch_in=64)
        self.block12=encoder_block2(channel_in=64,channel_out=128)
        self.block13=encoder_block3(channel_in=128,channel_out=256)
        self.block14=encoder_block4(channel_in=256,channel_out=512)

        # --- Transformer 分支 (上下文特征) ---
        self.block21 = transformer(image_size=(256, 256), axiss =96, channels=64,channel1=3,patch_size=(8, 8),mlp_axis=96*2,h=256,w=256) 
        self.block22 = transformer(image_size=(128, 128), axiss =192, channels=128,channel1=64,patch_size=(8, 8),mlp_axis=192*2,h=128,w=128)
        self.block23 = transformer(image_size=(64, 64),   axiss =384, channels=256,channel1=128,patch_size=(2, 2),mlp_axis=384*2,h=64,w=64)
        self.block24 = transformer(image_size=(32, 32),   axiss =768, channels=512,channel1=256,patch_size=(2, 2),mlp_axis=768*2,h=32,w=32)

        # --- 特征融合与增强模块 ---
        self.AF1=AF_block(ch_in=64,ch_out=64,ch_middle=8)
        self.AF2=AF_block(ch_in=128,ch_out=128,ch_middle=16)
        self.AF3=AF_block(ch_in=256,ch_out=256,ch_middle=32)
        self.AF4=AF_block(ch_in=512,ch_out=512,ch_middle=64)

        self.ALFE64=ALFEblock(ch_in=64,ch_out=64)
        self.ALFE128=ALFEblock(ch_in=128,ch_out=128)
        self.ALFE256=ALFEblock(ch_in=256,ch_out=256)
        self.ALFE512=ALFEblock(ch_in=512,ch_out=512)

        self.ER64=ER_block(ch_in=64,ch_out=64)
        self.ER128=ER_block(ch_in=128,ch_out=128)
        self.ER256=ER_block(ch_in=256,ch_out=256)
        self.ER512=ER_block(ch_in=512,ch_out=512)

        self.Deepest=MFE_block(ch_in=512,ch_out=512)

        # --- 解码器 ---
        self.Conv11=conv_block(1024,512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256,ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128,ch_out=64)

        self.Up1 = up_conv(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2D(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: 输入图像张量，形状 [B, 3, H, W]。
        :return: 输出分割图张量，形状 [B, output_ch, H, W]。
        """
        # --- Transformer 分支 ---
        t1 = self.block21(x)    # [B, 64, 128, 128]
        t2 = self.block22(t1)   # [B, 128, 64, 64]
        t3 = self.block23(t2)   # [B, 256, 32, 32]
        t4 = self.block24(t3)   # [B, 512, 16, 16]

        # --- 卷积分支 ---
        x_edge = edge_conv2d(x)
        x1 = self.block11(x_edge)
        x1_pool = self.pool(x1) # [B, 64, 128, 128]

        # --- 编码器与特征融合 ---
        # 尺度 1
        out1 =self.AF1(a=x1_pool,b=t1)
        out1=self.ALFE64(out1)
        out1=self.ER64(out1)

        x2 = self.block12(x1_pool) # [B, 128, 64, 64]
        
        # 尺度 2
        out2=self.AF2(a=x2,b=t2)
        out2=self.ALFE128(out2)
        out2=self.ER128(out2)
       
        x3 = self.block13(x2) # [B, 256, 32, 32]

        # 尺度 3
        out3=self.AF3(a=x3,b=t3)
        out3=self.ALFE256(out3)
        out3=self.ER256(out3)
        
        x4 = self.block14(x3) # [B, 512, 16, 16]

        # 尺度 4
        out4 =self.AF4(a=x4,b=t4)
        out4=self.ALFE512(out4)
        out4=self.ER512(out4)
        
        # --- 最深层 ---
        out5=self.Deepest(a=x4,b=t4) 
        d5 = paddle.concat(x=[out5, out4], axis=1)
        d5 = self.Conv11(d5)
        
        # --- 解码器 ---
        d4 = self.Up4(d5) # [B, 256, 32, 32]
        d4 = paddle.concat(x=[d4, out3], axis=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4) # [B, 128, 64, 64]
        d3 = paddle.concat(x=[d3,out2], axis=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3) # [B, 64, 128, 128]
        d2 = paddle.concat(x=[d2,out1], axis=1)
        d2 = self.Up_conv2(d2)
        
        d2 = self.Up1(d2) # [B, 32, 256, 256]
        
        d1 = self.Conv_1x1(d2) # [B, output_ch, 256, 256]

        return d1

# --- 模型测试 ---
IMAGE_SIZE = (256, 256)
num_classes = 2
network = Three_Net(img_ch=3, output_ch=num_classes)
model = paddle.Model(network)
model.summary((-1, 3,) + IMAGE_SIZE)
# FLOPs = paddle.flops(network, [1, 3, 256, 256], custom_ops= None, print_detail=True)
# print(FLOPs)
