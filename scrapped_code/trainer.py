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
from models.torch.YOLOv8 import multitask_loss

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

def compute_segmentation_metrics(pred, target) -> dict:
    dice = dice_score(pred, target)
    iou = iou_score(pred, target)
    precision, recall, f1 = compute_precision_recall_f1(pred, target)
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }

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
        
        # Enhanced metrics history for plotting - includes all segmentation metrics
        self.train_history = {
            'loss': [],
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'bbox_loss': [],
            'obj_loss': [],
            'cls_loss': [],
            'seg_loss': []
        }
        self.val_history = {
            'loss': [],
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'bbox_loss': [],
            'obj_loss': [],
            'cls_loss': [],
            'seg_loss': []
        }
        
        # Compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('compile', True):
            self.model = torch.compile(self.model)
            
    def train_epoch(self, epoch):
        self.model.train()
        metrics = defaultdict(float)
        seg_metrics = defaultdict(float)
        num_batches = len(self.train_loader)

        # Set epoch for distributed sampler if needed
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, (img, mask, bbox, label) in enumerate(self.train_loader):
            start = time.time()
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            bbox = bbox.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision autocast
            context = autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else nullcontext()
            with context:
                det_out, seg_out = self.model(img)
                loss, logs = multitask_loss(detection_pred=det_out,
                                      segmentation_pred=seg_out,
                                      bbox=bbox,
                                      label=label,
                                      mask=mask)
                
                # Compute segmentation metrics for training
                train_seg_metrics = compute_segmentation_metrics(seg_out, mask)

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
            
            # Accumulate metrics
            metrics['loss'] += logs['total']
            metrics['bbox_loss'] += logs['bbox']
            metrics['obj_loss'] += logs['objectness']
            metrics['cls_loss'] += logs['classification']
            metrics['seg_loss'] += logs['segmentation']
    
            
            # Accumulate segmentation metrics
            for key, value in train_seg_metrics.items():
                seg_metrics[key] += value
            
            batch_time = time.time() - start
            metrics['time'] += batch_time
            
            # Batch logging (reduced frequency to avoid spam)
            if batch_idx % 1 == 0:  # Reduced frequency from 1 to 10
                print(
                    f"\nEpoch {epoch}, Batch {batch_idx}/{num_batches}: \n"
                    f"  Loss: {logs['total']:.4f} | bbox: {logs['bbox']:.4f} | "
                    f"obj: {logs['objectness']:.4f} | cls: {logs['classification']:.4f} | "
                    f"seg: {logs['segmentation']:.4f} | dice: {train_seg_metrics['dice']:.4f} | "
                    f"iou: {train_seg_metrics['iou']:.4f} | LR: {self.scheduler.get_last_lr()[0]:.5f} | Batch time: {batch_time:.2f} \n"
                )

        # Average metrics over epoch
        for k in metrics:
            if k != 'time':
                metrics[k] /= num_batches
        
        for k in seg_metrics:
            seg_metrics[k] /= num_batches

        # Store in history
        self.train_history['loss'].append(metrics['loss'])
        self.train_history['bbox_loss'].append(metrics['bbox_loss'])
        self.train_history['obj_loss'].append(metrics['obj_loss'])
        self.train_history['cls_loss'].append(metrics['cls_loss'])
        self.train_history['seg_loss'].append(metrics['seg_loss'])
        self.train_history['dice'].append(seg_metrics['dice'])
        self.train_history['iou'].append(seg_metrics['iou'])
        self.train_history['precision'].append(seg_metrics['precision'])
        self.train_history['recall'].append(seg_metrics['recall'])
        self.train_history['f1'].append(seg_metrics['f1'])

        print(
            f"\nTraining Epoch {epoch} Summary:\n"
            f"Loss: {metrics['loss']:.4f} | Dice: {seg_metrics['dice']:.4f} | "
            f"IoU: {seg_metrics['iou']:.4f} | F1: {seg_metrics['f1']:.4f}"
        )

        return {**metrics, **seg_metrics}
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        metrics = defaultdict(float)
        seg_metrics = defaultdict(float)
        num_batches = len(self.val_loader)

        for batch_idx, (img, mask, bbox, label) in enumerate(self.val_loader):
            start = time.time()
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            bbox = bbox.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            with autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else nullcontext():
                det_out, seg_out = self.model(img)
                _, logs = multitask_loss(detection_pred=det_out,
                                      segmentation_pred=seg_out,
                                      bbox=bbox,
                                      label=label,
                                      mask=mask)
                
                val_seg_metrics = compute_segmentation_metrics(seg_out, mask)

            # Accumulate metrics
            metrics['loss'] += logs['total']
            metrics['bbox_loss'] += logs['bbox']
            metrics['obj_loss'] += logs['objectness']
            metrics['cls_loss'] += logs['classification']
            metrics['seg_loss'] += logs['segmentation']
            
            # Accumulate segmentation metrics
            for key, value in val_seg_metrics.items():
                seg_metrics[key] += value
            
            batch_time = time.time() - start
            metrics['time'] += batch_time

            # Reduced logging frequency
            if batch_idx % 1 == 0:  # Reduced frequency from 1 to 10
                print(
                    f"\nEpoch {epoch}, Validation batch {batch_idx}/{num_batches}: \n"
                    f"  Loss: {logs['total']:.4f} | bbox: {logs['bbox']:.4f} | "
                    f"obj: {logs['objectness']:.4f} | cls: {logs['classification']:.4f} | "
                    f"seg: {logs['segmentation']:.4f} | dice: {val_seg_metrics['dice']:.4f} | "
                    f"iou: {val_seg_metrics['iou']:.4f} | Batch time: {batch_time:.2f} \n"
                )

        # Average metrics
        for k in metrics:
            if k != 'time':
                metrics[k] /= num_batches
        
        for k in seg_metrics:
            seg_metrics[k] /= num_batches

        # Store in history
        self.val_history['loss'].append(metrics['loss'])
        self.val_history['bbox_loss'].append(metrics['bbox_loss'])
        self.val_history['obj_loss'].append(metrics['obj_loss'])
        self.val_history['cls_loss'].append(metrics['cls_loss'])
        self.val_history['seg_loss'].append(metrics['seg_loss'])
        self.val_history['dice'].append(seg_metrics['dice'])
        self.val_history['iou'].append(seg_metrics['iou'])
        self.val_history['precision'].append(seg_metrics['precision'])
        self.val_history['recall'].append(seg_metrics['recall'])
        self.val_history['f1'].append(seg_metrics['f1'])

        print(
            f"\nValidation Epoch {epoch} Summary:\n"
            f"Loss: {metrics['loss']:.4f} | Dice: {seg_metrics['dice']:.4f} | "
            f"IoU: {seg_metrics['iou']:.4f} | F1: {seg_metrics['f1']:.4f}"
        )

        return {**metrics, **seg_metrics}

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
            'current_metrics': metrics,
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
        """Plot comprehensive training metrics with better organization"""
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots with better organization
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle('Training Metrics Dashboard', fontsize=16)
            
            # Define metrics to plot with their positions
            metrics_layout = [
                # Row 1: Loss metrics
                ('loss', 0, 0, 'Total Loss'),
                ('seg_loss', 0, 1, 'Segmentation Loss'),
                ('bbox_loss', 0, 2, 'BBox Loss'),
                ('obj_loss', 0, 3, 'Objectness Loss'),
                # Row 2: Segmentation metrics
                ('dice', 1, 0, 'Dice Score'),
                ('iou', 1, 1, 'IoU Score'),
                ('f1', 1, 2, 'F1 Score'),
                ('cls_loss', 1, 3, 'Classification Loss'),
                # Row 3: Precision/Recall and comparison
                ('precision', 2, 0, 'Precision'),
                ('recall', 2, 1, 'Recall'),
                ('', 2, 2, ''),  # Empty for custom plot
                ('', 2, 3, '')   # Empty for custom plot
            ]
            
            for metric, row, col, title in metrics_layout:
                if metric and metric in self.train_history and len(self.train_history[metric]) > 0:
                    epochs = range(1, len(self.train_history[metric]) + 1)
                    axes[row, col].plot(epochs, self.train_history[metric], 'b-', label='Train', linewidth=2, alpha=0.8)
                    axes[row, col].plot(epochs, self.val_history[metric], 'r-', label='Val', linewidth=2, alpha=0.8)
                    axes[row, col].set_title(title, fontsize=12, fontweight='bold')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                    
                    # Add value annotations for the last epoch
                    if len(epochs) > 0:
                        last_train = self.train_history[metric][-1]
                        last_val = self.val_history[metric][-1]
                        axes[row, col].annotate(f'{last_train:.3f}', 
                                              xy=(epochs[-1], last_train), 
                                              xytext=(5, 5), textcoords='offset points',
                                              fontsize=8, color='blue')
                        axes[row, col].annotate(f'{last_val:.3f}', 
                                              xy=(epochs[-1], last_val), 
                                              xytext=(5, -15), textcoords='offset points',
                                              fontsize=8, color='red')
            
            # Custom plot 1: Dice vs IoU comparison
            if len(self.val_history['dice']) > 0 and len(self.val_history['iou']) > 0:
                epochs = range(1, len(self.val_history['dice']) + 1)
                axes[2, 2].plot(epochs, self.val_history['dice'], 'g-', label='Dice', linewidth=2)
                axes[2, 2].plot(epochs, self.val_history['iou'], 'orange', label='IoU', linewidth=2)
                axes[2, 2].set_title('Dice vs IoU (Validation)', fontsize=12, fontweight='bold')
                axes[2, 2].set_xlabel('Epoch')
                axes[2, 2].legend()
                axes[2, 2].grid(True, alpha=0.3)
            
            # Custom plot 2: Learning rate over time
            if hasattr(self, 'scheduler'):
                # Note: This is approximate since we don't store LR history
                epochs = range(1, len(self.train_history['loss']) + 1)
                axes[2, 3].set_title('Training Progress Summary', fontsize=12, fontweight='bold')
                axes[2, 3].text(0.1, 0.8, f"Best Dice: {max(self.val_history['dice']) if self.val_history['dice'] else 0:.4f}", 
                               transform=axes[2, 3].transAxes, fontsize=10)
                axes[2, 3].text(0.1, 0.7, f"Best IoU: {max(self.val_history['iou']) if self.val_history['iou'] else 0:.4f}", 
                               transform=axes[2, 3].transAxes, fontsize=10)
                axes[2, 3].text(0.1, 0.6, f"Best F1: {max(self.val_history['f1']) if self.val_history['f1'] else 0:.4f}", 
                               transform=axes[2, 3].transAxes, fontsize=10)
                axes[2, 3].text(0.1, 0.5, f"Lowest Loss: {min(self.val_history['loss']) if self.val_history['loss'] else 0:.4f}", 
                               transform=axes[2, 3].transAxes, fontsize=10)
                axes[2, 3].text(0.1, 0.4, f"Epochs: {len(epochs)}", 
                               transform=axes[2, 3].transAxes, fontsize=10)
                axes[2, 3].set_xlim(0, 1)
                axes[2, 3].set_ylim(0, 1)
                axes[2, 3].axis('off')
            
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

    def get_best_metrics(self):
        """Get best achieved metrics across all epochs"""
        best_metrics = {}
        
        for metric_name in ['dice', 'iou', 'f1', 'precision', 'recall']:
            if metric_name in self.val_history and self.val_history[metric_name]:
                best_metrics[f'best_{metric_name}'] = max(self.val_history[metric_name])
                best_metrics[f'best_{metric_name}_epoch'] = self.val_history[metric_name].index(best_metrics[f'best_{metric_name}']) + 1
        
        if 'loss' in self.val_history and self.val_history['loss']:
            best_metrics['best_loss'] = min(self.val_history['loss'])
            best_metrics['best_loss_epoch'] = self.val_history['loss'].index(best_metrics['best_loss']) + 1
        
        return best_metrics



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