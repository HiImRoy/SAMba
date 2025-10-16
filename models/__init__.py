# Author: Roy
# Copyright (c) Roy. All rights reserved.

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
# [Roy] 从新的decoder.py中导入损失函数
from models.decoder import bce_dice
# [Roy] 导入主模型
from mmcls.SAVSS_dev.models.SAVSS.SAMbaCrack import SAMbaCrack

def build_model(args):
    """
    功能: 根据传入的参数构建SAMbaCrack模型和损失函数。
    [Roy] 移除了对旧模型(如UNetBaseline)的支持，现在只构建SAMbaCrack。
    """
    device = torch.device(args.device)

    # [Roy] 直接实例化SAMbaCrack模型，不再需要根据model_name判断
    model = SAMbaCrack(args=args)

    # 损失函数在decoder.py中定义
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion
