'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Iterable
import torch
import time
from tqdm import tqdm
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, args = None, logger = None):
    model.train()
    criterion.train()

    epoch_losses = []
    pbar = tqdm(total=len(data_loader), desc=f"Epoch {epoch} Training")
    for i, data in enumerate(data_loader):
        if isinstance(data, (list, tuple)):
            samples, targets = data
            samples = samples.to(torch.device(args.device))
            targets = targets.to(torch.device(args.device))
        else:
            samples = data['image'].to(torch.device(args.device))
            targets = data['label'].to(torch.device(args.device))

        output = model(samples)
        
        # --- [FIXED] Add channel dimension to target mask to match model output --- #
        # output shape: [B, 1, H, W], original targets shape: [B, H, W]
        # We need to make targets shape [B, 1, H, W] for the loss function.
        loss_final = criterion(output, targets.unsqueeze(1).float())

        epoch_losses.append(loss_final.item())

        pbar.set_description(f"Epoch {epoch} | Batch Loss: {loss_final.item():.4f}")
        pbar.update(1)
        
        optimizer.zero_grad()
        loss_final.backward()

        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optimizer.step()

    pbar.close()

    avg_epoch_loss = np.mean(epoch_losses)
    lr = optimizer.param_groups[0]['lr']
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

    logger.info(f"time -> {cur_time} | Epoch -> {epoch} | Average Train Loss -> {avg_epoch_loss:.4f} | lr -> {lr}")

    return {'loss': avg_epoch_loss}
