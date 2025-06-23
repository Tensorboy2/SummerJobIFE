import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math
from contextlib import nullcontext
import time
from datetime import datetime
import os


@torch.jit.script
def dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    """Computes Dice and IoU from binary predictions and targets."""
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    iou_union = (pred + target - pred * target).sum()

    dice = 2 * intersection / (union + eps)
    iou = intersection / (iou_union + eps)

    return dice.item(), iou.item()

class EfficientTrainer:
    """Production-ready trainer with all optimizations"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Mixed precision setup
        self.scaler = GradScaler(device=self.device)
        self.use_amp = config.get('use_amp', True)
        
        # Optimizer with proper weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.05),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = config['num_epochs']*(len(train_loader))
        # Learning rate scheduler
        def lr_lambda(step):
            warmup_steps = config['warmup_steps']
            decay = config['decay']
            if step < warmup_steps:
                return (step + 1)/ warmup_steps
            else:
                decay_epochs = total_steps - warmup_steps
                decay_progress = (step - warmup_steps) / decay_epochs
                if decay=="linear":
                    return max(0.0, 1.0 - decay_progress)
                elif decay=="cosine":
                    return 0.5 * (1 + math.cos(math.pi * decay_progress))
                else:
                    return 1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,lr_lambda
        )
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('compile', True):
            self.model = torch.compile(self.model)
            
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_class = 0
        num_batches = len(self.train_loader)
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            start = time.time()
            data, target = data.to(self.device, non_blocking=True), \
                          target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision context
            with autocast(self.device) if self.use_amp else nullcontext():
                seg_out, class_out = self.model(data)
                # print(seg_out.shape)
                # Compute losses
                bce_loss = F.binary_cross_entropy_with_logits(seg_out, target)
                dice_loss_val, iou = dice_iou(seg_out, target)

                # Binary classification target: 1 if any foreground exists
                has_mask = (target.sum(dim=(1, 2, 3)) > 0).float()
                class_loss = F.binary_cross_entropy_with_logits(class_out.squeeze(), has_mask)

                # Combine
                loss = bce_loss + dice_loss_val + 0.1 * class_loss
            
            # Backward pass with mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
            total_loss += loss.item()
            total_dice += dice_loss_val
            total_iou += iou
            total_class += class_loss
            stop = time.time()
            # Logging every N steps
            # if batch_idx % 1 == 0:
            print(f'\nEpoch {epoch}, Batch {batch_idx}/{num_batches},\n' 
                  f'Loss: {loss.item():.6f}, LR: {self.scheduler.get_last_lr()[0]:.5f},\n'
                  f'F1: {dice_loss_val:.6f}, Iou: {iou:.6f},\n'
                  f'Class loss: {class_loss:.3f}\n'
                  f'Batch time: {stop-start:.2f} seconds\n')
                
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches

        print(f'\nTraining Summary: Epoch: {epoch}, Avg Loss: {avg_loss:.6f}, '
            f'Avg Dice: {avg_dice:.6f}, Avg IoU: {avg_iou:.6f}\n')
        return avg_loss, avg_dice, avg_iou
    
    @torch.no_grad()
    def validate(self,epoch):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        num_batches = len(self.val_loader)

        for batch_idx, (data, target) in enumerate(self.val_loader):
            start = time.time()

            data, target = data.to(self.device, non_blocking=True), \
                        target.to(self.device, non_blocking=True)

            with autocast(self.device) if self.use_amp else nullcontext():
                seg_out, class_out = self.model(data)
                # print(seg_out.shape)
                # Compute losses
                bce_loss = F.binary_cross_entropy_with_logits(seg_out, target)
                dice_loss_val, iou = dice_iou(seg_out, target)

                # Binary classification target: 1 if any foreground exists
                has_mask = (target.sum(dim=(1, 2, 3)) > 0).float()
                class_loss = F.binary_cross_entropy_with_logits(class_out.squeeze(), has_mask)

                # Combine
                loss = bce_loss + dice_loss_val + 0.1 * class_loss

            total_loss += loss.item()
            total_dice += dice_loss_val
            total_iou += iou
            stop = time.time()

            print(f'Epoch {epoch}, Validation Batch {batch_idx}/{num_batches},' 
                  f'Loss: {loss.item():.6f},'
                  f'F1: {dice_loss_val:.6f}, Iou: {iou:.6f},'
                  f'Batch time: {stop-start:.2f} seconds')
            # print(f'Validation Batch {batch_idx}/{num_batches}, '
            #     f'Loss: {loss.item():.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}')

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        print(f'\nValidation Summary: Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, '
            f'Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}\n')

        return avg_loss, avg_dice, avg_iou

    
    def save_checkpoint(self, epoch, loss, iou, dice, folderpath):
        """Efficient checkpointing"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'iou': iou,
            'dice': dice,
            'config': self.config
        }

        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d_%H:%M:%S")

        # Async saving to avoid blocking training
        torch.save(checkpoint, os.path.join(folderpath,formatted))

# if __name__ == '__main__':
#     trainer = EfficientTrainer()