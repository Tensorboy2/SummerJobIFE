import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math
from contextlib import nullcontext
import time
from datetime import datetime
import os
import json
from collections import defaultdict
from models.torch.yolo_utils.loss import multitask_loss

@torch.jit.script
def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes IoU score between predicted and target masks.

    Args:
        pred (Tensor): predicted mask of shape (N, 1, H, W) with values in [0, 1] or logits
        target (Tensor): ground truth mask of shape (N, 1, H, W) with values in {0, 1}
        threshold (float): threshold to binarize predictions
        eps (float): small constant for numerical stability

    Returns:
        Tensor: IoU score averaged over batch
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = torch.sum(pred_bin * target, dim=(1, 2, 3))
    union = torch.sum(pred_bin + target, dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()

@torch.jit.script
def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes Dice score between predicted and target masks.

    Args:
        pred (Tensor): predicted mask of shape (N, 1, H, W) with values in [0, 1] or logits
        target (Tensor): ground truth mask of shape (N, 1, H, W) with values in {0, 1}
        threshold (float): threshold to binarize predictions
        eps (float): small constant for numerical stability

    Returns:
        Tensor: Dice score averaged over batch
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = torch.sum(pred_bin * target, dim=(1, 2, 3))
    union = torch.sum(pred_bin, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

@torch.jit.script
def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for training (uses sigmoid activation)
    
    Args:
        pred (Tensor): predicted logits of shape (N, 1, H, W)
        target (Tensor): ground truth mask of shape (N, 1, H, W) with values in {0, 1}
        eps (float): small constant for numerical stability
    
    Returns:
        Tensor: Dice loss (1 - dice_score)
    """
    pred_sigmoid = torch.sigmoid(pred)
    pred_flat = pred_sigmoid.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2. * intersection + eps) / (union + eps)
    
    return 1 - dice

@torch.jit.script
def compute_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision, recall, and F1 score"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    # Flatten for computation
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    
    return precision, recall, f1

class EfficientTrainer:
    """Production-ready trainer with all optimizations"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss combination weights
        self.bce_weight = config.get('bce_weight', 0.5)
        self.dice_weight = config.get('dice_weight', 0.5)
        
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
        
        # Metrics history for plotting
        self.train_history = {
            'loss': [], 'bce': [], 'dice_loss': [], 'dice_score': [], 'iou': [],
            'precision': [], 'recall': [], 'f1': []
        }
        self.val_history = {
            'loss': [], 'bce': [], 'dice_loss': [], 'dice_score': [], 'iou': [],
            'precision': [], 'recall': [], 'f1': []
        }
        
        # Compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('compile', True):
            self.model = torch.compile(self.model)
            
    def train_epoch(self, epoch):
        self.model.train()
        metrics = defaultdict(float)
        num_batches = len(self.train_loader)

        # Set epoch for distributed sampler if needed
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, (img, mask, bbox, label) in enumerate(self.train_loader):
            start = time.time()
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            bbox =bbox.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision autocast
            context = autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else nullcontext()
            with context:
                det_out, seg_out = self.model(img)

                # Compute losses
                # bce_loss = F.binary_cross_entropy_with_logits(seg_out, target)
                # dice_loss_val = dice_loss(seg_out, target)
                
                # Combined loss
                # loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss_val
                loss, logs = multitask_loss(det_pred=det_out,
                                      seg_pred=seg_out,
                                      bbox=bbox,
                                      label=label,
                                      mask=mask)
                
                # Compute metrics (for monitoring)
                # with torch.no_grad():
                #     dice_score_val = dice_score(seg_out, target)
                #     iou_val = iou_score(seg_out, target)
                #     precision, recall, f1 = compute_precision_recall_f1(seg_out, target)

            # Backpropagation
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

            self.scheduler.step()
            print(loss.item())
            # Accumulate metrics
            metrics['loss'] += loss.item()
            # metrics['bce'] += bce_loss.item()
            # metrics['dice_loss'] += dice_loss_val.item()
            # metrics['dice_score'] += dice_score_val.item()
            # metrics['iou'] += iou_val.item()
            # metrics['precision'] += precision.item()
            # metrics['recall'] += recall.item()
            # metrics['f1'] += f1.item()
            
            batch_time = time.time() - start
            metrics['time'] += batch_time
            
            # Batch logging (reduced frequency to avoid spam)
            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                    f"Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.5f}"
                )

        # Average metrics over epoch
        for k in metrics:
            if k != 'time':
                metrics[k] /= num_batches

        # Store in history
        for key in self.train_history.keys():
            self.train_history[key].append(metrics[key])

        print(
            f"\nTraining Epoch {epoch} Summary:\n"
            f"Loss: {metrics['loss']:.4f} "
        )

        return metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = len(self.val_loader)

        for batch_idx, (img, mask, bbox, label) in enumerate(self.val_loader):
            start = time.time()
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else nullcontext():
                seg_out = self.model(data)

                # Compute losses and metrics
                bce_loss = F.binary_cross_entropy_with_logits(seg_out, target)
                dice_loss_val = dice_loss(seg_out, target)
                loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss_val
                
                dice_score_val = dice_score(seg_out, target)
                iou_val = iou_score(seg_out, target)
                precision, recall, f1 = compute_precision_recall_f1(seg_out, target)

            # Accumulate metrics
            metrics['loss'] += loss.item()
            metrics['bce'] += bce_loss.item()
            metrics['dice_loss'] += dice_loss_val.item()
            metrics['dice_score'] += dice_score_val.item()
            metrics['iou'] += iou_val.item()
            metrics['precision'] += precision.item()
            metrics['recall'] += recall.item()
            metrics['f1'] += f1.item()
            
            batch_time = time.time() - start
            metrics['time'] += batch_time

            # Reduced logging frequency
            if batch_idx % 1 == 0:
                print(
                    f"Val Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                    f"Loss: {loss.item():.4f} | Dice: {dice_score_val.item():.4f} | "
                    f"IoU: {iou_val.item():.4f}"
                )

        # Average metrics
        for k in metrics:
            if k != 'time':
                metrics[k] /= num_batches

        # Store in history
        for key in self.val_history.keys():
            self.val_history[key].append(metrics[key])

        print(
            f"\nValidation Epoch {epoch} Summary:\n"
            f"Loss: {metrics['loss']:.4f} | BCE: {metrics['bce']:.4f} | "
            f"Dice Loss: {metrics['dice_loss']:.4f} | Dice Score: {metrics['dice_score']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
        )

        return metrics

    def save_checkpoint(self, epoch, metrics, folderpath):
        """Enhanced checkpointing with complete metrics history"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
            'loss_weights': {
                'bce_weight': self.bce_weight,
                'dice_weight': self.dice_weight
            }
        }

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folderpath, f'checkpoint_epoch{epoch}_{now}.pt')
        torch.save(checkpoint, filename)
        
        # Also save metrics as JSON for easy analysis
        metrics_filename = os.path.join(folderpath, f'metrics_epoch{epoch}_{now}.json')
        metrics_data = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'epoch': epoch,
            'loss_weights': {
                'bce_weight': self.bce_weight,
                'dice_weight': self.dice_weight
            }
        }
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Checkpoint saved to: {filename}")
        print(f"Metrics saved to: {metrics_filename}")

    def plot_metrics(self, save_path=None):
        """Plot comprehensive training metrics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            metrics_to_plot = ['loss', 'dice_score', 'iou', 'precision', 'recall', 'f1', 'bce', 'dice_loss']
            
            for i, metric in enumerate(metrics_to_plot):
                row = i // 4
                col = i % 4
                
                if metric in self.train_history and len(self.train_history[metric]) > 0:
                    epochs = range(1, len(self.train_history[metric]) + 1)
                    axes[row, col].plot(epochs, self.train_history[metric], 'b-', label='Train', linewidth=2)
                    axes[row, col].plot(epochs, self.val_history[metric], 'r-', label='Val', linewidth=2)
                    axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Metrics plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore metrics history if available
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
            self.val_history = checkpoint['val_history']
        
        # Restore loss weights if available
        if 'loss_weights' in checkpoint:
            self.bce_weight = checkpoint['loss_weights']['bce_weight']
            self.dice_weight = checkpoint['loss_weights']['dice_weight']
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

# Example training loop
def train_model(trainer, num_epochs, checkpoint_dir, save_every=5, plot_every=10):
    """Complete training loop with checkpointing and plotting"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_dice = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = trainer.train_epoch(epoch)
        
        # Validate
        val_metrics = trainer.validate(epoch)
        
        # Save checkpoint for best model
        if val_metrics['dice_score'] > best_dice:
            best_dice = val_metrics['dice_score']
            trainer.save_checkpoint(epoch, val_metrics, checkpoint_dir)
            print(f"New best Dice score: {best_dice:.4f}")
        
        # Regular checkpointing
        if epoch % save_every == 0:
            trainer.save_checkpoint(epoch, val_metrics, checkpoint_dir)
        
        # Plot metrics
        if epoch % plot_every == 0:
            plot_path = os.path.join(checkpoint_dir, f'metrics_epoch_{epoch}.png')
            trainer.plot_metrics(save_path=plot_path)

# Example usage:
if __name__ == '__main__':
    # Example config
    config = {
        'lr': 1e-4,
        'num_epochs': 100,
        'warmup_steps': 1000,
        'decay': 'cosine',
        'weight_decay': 0.05,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
        'bce_weight': 0.4,    # Adjust based on your needs
        'dice_weight': 0.6,   # Higher weight for spatial coherence
    }
    
    # trainer = EfficientTrainer(model, train_loader, val_loader, device, config)
    # train_model(trainer, num_epochs=100, checkpoint_dir='./checkpoints')