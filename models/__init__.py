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
from mmcls.SAVSS_dev.models.SAVSS.SAMbaCrack import SAMbaCrack

def build_model(args):
    """
    Builds the model and criterion based on the provided arguments.
    This function now correctly instantiates the SAMbaCrack model.
    """
    device = torch.device(args.device)
    
    # 1. Instantiate the correct SAMbaCrack model
    model = SAMbaCrack(args=args)
    
    # 2. The loss function is defined in decoder.py, which we can reuse.
    criterion = bce_dice(args) 
    criterion.to(device)

    return model, criterion
