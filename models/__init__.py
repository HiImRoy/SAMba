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
    This function now supports switching between SAMbaCrack and UNetBaseline.
    """
    device = torch.device(args.device)
    model = None

    # --- Build model based on model_name ---
    if args.model_name == 'UNetBaseline':
        model = UNetBaseline(n_channels=3, n_classes=1)
        # Quick check of parameter count
        print(f"--- Built UNetBaseline model with ~{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters. ---")
    elif args.model_name == 'SAMbaCrack':
        model = SAMbaCrack(args=args)
    else:
        raise ValueError(f"Model '{args.model_name}' not recognized.")

    # The loss function is defined in decoder.py, which we can reuse.
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion
