# --- ROBUST IMPORT FIX: Manually add project root to sys.path ---
import sys
import os

# This calculates the absolute path to the project root (SAMba) and adds it to Python's path.
# It navigates up from the current file's directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END ROBUST IMPORT FIX ---

"""Model factory."""

import torch
from models.decoder import bce_dice

# [FIXED] 导入我们最终组装好的、正确的 SAMbaCrack 模型
from mmcls.SAVSS_dev.models.SAVSS.SAMbaCrack import SAMbaCrack

# 保留旧的 UNetBaseline 以便进行比较
from models.unet_baseline import UNetBaseline

def build_model(args):
    """
    构建模型和损失函数。
    该函数现在会根据 model_name 构建我们最终确定的 SAMbaCrack-AF 架构。
    """
    device = torch.device(args.device)
    model = None

    # --- 根据 model_name 构建模型 ---
    if args.model_name == 'UNetBaseline':
        model = UNetBaseline(n_channels=3, n_classes=1)
        print(f"--- 已构建 UNetBaseline 模型，约 {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M 参数。 ---")
    
    # [FIXED] 当模型名称为 SAMbaCrack 时，实例化我们全新的、模块化的 SAMbaCrack 模型
    elif args.model_name == 'SAMbaCrack':
        # 注意：现在实例化的类是 SAMbaCrack
        model = SAMbaCrack(args=args)
    
    else:
        raise ValueError(f"无法识别模型 \'{args.model_name}\'。")

    # 损失函数保持不变，仍然使用 bce_dice 组合损失
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion
