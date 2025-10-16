'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
from tqdm import tqdm
import util.misc as utils

# --- [RESTORED] Added back 'training_stage' argument ---
def train_one_epoch(model, criterion, data_loader, optimizer, epoch, args, log, training_stage):
    model.train()
    criterion.train()
    
    total_loss = 0
    
    # TQDM progress bar setup
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} Training")

    for data in progress_bar:
        samples = data["image"].to(torch.device(args.device))
        targets = data["label"].to(torch.device(args.device))

        # Pass the training stage to the model
        outputs = model(samples, stage=training_stage)
        
        loss = criterion(outputs, targets.float())
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update TQDM description with current average loss
        progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

    avg_loss = total_loss / len(data_loader)
    lr = optimizer.param_groups[0]['lr']
    log.info(f"Epoch {epoch} Training - Average Loss: {avg_loss:.4f} | lr: {lr:.6f}")
    
    return {"loss": avg_loss}
