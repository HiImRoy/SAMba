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

# Modified function to accept a scheduler and perform gradient clipping and per-batch stepping
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, # ADDED: for per-batch updates
                    epoch: int, args = None, logger = None):
    model.train()
    criterion.train()

    epoch_losses = []
    # Use len(data_loader) which is now correct after the fix in datasets/__init__.py
    pbar = tqdm(total=len(data_loader), desc=f"Epoch {epoch} Training")
    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        output = model(samples)
        loss_final = criterion(output, targets.float())

        epoch_losses.append(loss_final.item())

        # Update progress bar with current batch loss and learning rate
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Epoch {epoch} | Batch Loss: {loss_final.item():.4f} | LR: {current_lr:.2e}")
        pbar.update(1)
        
        optimizer.zero_grad()
        loss_final.backward()

        # ADDED: Gradient Clipping
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        optimizer.step()

        # ADDED: Per-batch scheduler step for OneCycleLR
        lr_scheduler.step()

    pbar.close()

    # Calculate and log average loss for the epoch
    avg_epoch_loss = np.mean(epoch_losses)
    # The final LR of the epoch will be very low with OneCycleLR, so logging it is fine.
    final_lr = optimizer.param_groups[0]['lr']
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

    logger.info(f"time -> {cur_time} | Epoch -> {epoch} | Average Train Loss -> {avg_epoch_loss:.4f} | Final lr -> {final_lr:.6f}")

    # Return stats for logging in main.py
    return {'loss': avg_epoch_loss}
