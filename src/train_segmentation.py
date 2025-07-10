import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import math
import time
from typing import Dict
import os

@torch.jit.script
def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = (pred > threshold).float()
    target = target.float()

    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tn = torch.sum((1 - pred) * (1 - target))

    eps = 1e-6

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
    }

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        if device == 'string':
            # If device is a string, convert to torch.device
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.config = config

        # Set up output directory for checkpoints and metrics
        self.output_dir = os.path.join('checkpoints', 'segmentation', self.config['specific_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, f"pretrained_{self.config['specific_name']}.pt")

        self.use_amp = config.get('use_amp', True)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.05),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = self._build_scheduler()
        self.loss = {'train': [],'val': []}

        # Store epoch-averaged metrics
        self.val_metrics = {"precision": [], "recall": [], "dice": [], "iou": []}

        if hasattr(torch, 'compile') and config.get('compile', True):
            self.model = torch.compile(self.model)

    def _build_scheduler(self):
        total_steps = self.config['num_epochs'] * len(self.train_loader)
        warmup_steps = self.config.get('warmup_steps', 100)
        decay_type = self.config.get('decay', 'linear')

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            if decay_type == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            elif decay_type == "linear":
                return max(0.0, 1.0 - progress)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _step(self, img, mask, training=True):
        img = img.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        context = autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp) \
            if self.use_amp else nullcontext()
        with context:
            pred = self.model(img)
            loss = F.binary_cross_entropy_with_logits(pred, mask)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

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
            return loss.item()
        else:
            # For validation, return both loss and metrics
            # Apply sigmoid to get probabilities for metrics calculation
            pred_probs = torch.sigmoid(pred)
            metrics = compute_segmentation_metrics(pred_probs, mask)
            return loss.item(), metrics

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start = time.time()
        
        for batch_idx, (img, mask) in enumerate(self.train_loader):
            loss = self._step(img, mask, training=True)
            total_loss += loss
            
        tot_time = time.time() - start
        avg_loss = total_loss / len(self.train_loader)
        self.loss['train'].append(avg_loss)
        
        print(f"\n[Train Epoch {epoch}] Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.5f}, Epoch time: {tot_time:.2f}\n")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        # Accumulate metrics for averaging
        batch_metrics = {"precision": [], "recall": [], "dice": [], "iou": []}
        
        start = time.time()
        for batch_idx, (img, mask) in enumerate(self.val_loader):
            loss, metrics = self._step(img, mask, training=False)
            total_loss += loss
            
            # Collect metrics from this batch
            for key, value in metrics.items():
                batch_metrics[key].append(value)
                
        tot_time = time.time() - start
        
        # Calculate epoch averages
        avg_loss = total_loss / len(self.val_loader)
        self.loss['val'].append(avg_loss)
        
        # Calculate and store epoch-averaged metrics
        epoch_metrics = {}
        for key, values in batch_metrics.items():
            avg_metric = sum(values) / len(values)
            epoch_metrics[key] = avg_metric
            self.val_metrics[key].append(avg_metric)
        
        print(f"\n[Validation Epoch {epoch}] Loss: {avg_loss:.4f}, Epoch time: {tot_time:.2f}")
        print(f"Metrics - Precision: {epoch_metrics['precision']:.4f}, Recall: {epoch_metrics['recall']:.4f}, "
              f"Dice: {epoch_metrics['dice']:.4f}, IoU: {epoch_metrics['iou']:.4f}\n")

    def save_checkpoint(self, epoch=None, is_best=False):
        """
        Save only the best model weights (state_dict).
        """
        if is_best:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'config': self.config
            }
            fname = f"best_model_{self.config['specific_name']}.pt"
            path = os.path.join(self.output_dir, fname)
            torch.save(checkpoint, path)
            return path

    def save_encoder_checkpoint(self, epoch=None, is_best=False):
        """
        Save encoder and decoder weights. Optionally include epoch and best flag in filename.
        """
        fname = f"encoder_{self.config['specific_name']}"
        if epoch is not None:
            fname += f"_epoch{epoch}"
        if is_best:
            fname += "_best"
        fname += ".pt"
        path = os.path.join(self.output_dir, fname)
        checkpoint = {
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
            'epoch': epoch,
            'config': self.config
        }
        torch.save(checkpoint, path)
        return path

    def train(self):
        import json
        best_loss = float('inf')
        best_epoch = None
        metrics_path = os.path.join(self.output_dir, f'{self.config["specific_name"]}_metrics.pt')
        config_path = os.path.join(self.output_dir, 'config.json')
        # Save config for reproducibility
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        for epoch in range(1, self.config["num_epochs"] + 1):
            self.train_epoch(epoch)
            self.validate(epoch)
            val_loss = self.loss['val'][epoch-1]
            # Save metrics after each epoch
            torch.save({'loss': self.loss, 'val_metrics': self.val_metrics}, metrics_path)
            # Only save best model weights
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_ckpt = self.save_checkpoint(epoch=epoch, is_best=True)
                best_encoder = self.save_encoder_checkpoint(epoch=epoch, is_best=True)
                print(f"\nNew best model saved! Loss: {best_loss:.4f} (epoch {epoch})\nSaved: {best_ckpt}\nEncoder: {best_encoder}\n")

        # Save final summary
        summary = {
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'metrics': self.loss,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        summary_path = os.path.join(self.output_dir, f'{self.config["specific_name"]}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining complete. Best loss: {best_loss:.4f} (epoch {best_epoch})\nSummary saved to {summary_path}\n")


if __name__ == "__main__":
    print("\nTesting SegmentationTrainer with ViT and ConvNeXtV2 segmentation models:")
    from torch.utils.data import DataLoader, TensorDataset
    from models.torch.vit import create_vit_segmentation
    from models.torch.convnextv2rms import create_convnextv2_segmentation

    # Dummy dataset: 10 samples, 12 channels, 256x256, and binary masks
    dummy_imgs = torch.randn(4, 12, 256, 256)
    dummy_masks = (torch.rand(4, 1, 256, 256) > 0.5).float()
    dummy_loader = DataLoader(TensorDataset(dummy_imgs, dummy_masks), batch_size=1)

    # Minimal config
    config = {
        'specific_name': 'test',
        'lr': 1e-4,
        'num_epochs': 1,
        'compile': False
    }
    device = torch.device('cpu')

    # Import SegmentationTrainer explicitly (fix NameError)
    from __main__ import SegmentationTrainer

    # Test ViT Segmentation with SegmentationTrainer
    print("\n--- ViT Segmentation Trainer ---")
    vit_seg = create_vit_segmentation(size='base', in_channels=12, num_classes=1, patch_size=16)
    trainer_vit = SegmentationTrainer(vit_seg, dummy_loader, dummy_loader, device, config)
    trainer_vit.train_epoch(1)
    trainer_vit.validate(1)

    # Test ConvNeXtV2 Segmentation with SegmentationTrainer
    print("\n--- ConvNeXtV2 Segmentation Trainer ---")
    convnext_seg = create_convnextv2_segmentation(size='base', in_chans=12, num_classes=1)
    trainer_convnext = SegmentationTrainer(convnext_seg, dummy_loader, dummy_loader, device, config)
    trainer_convnext.train_epoch(1)
    trainer_convnext.validate(1)