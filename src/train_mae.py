import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from collections import defaultdict
import math
import time
import os


class MAETrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Set up output directory for checkpoints and metrics
        self.output_dir = os.path.join('checkpoints', 'mae', self.config['specific_name'])
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

    def _step(self, img, training=True):
        # Unpack if batch is a tuple/list (e.g., from TensorDataset)
        if isinstance(img, (tuple, list)):
            img = img[0]
        img = img.to(self.device, non_blocking=True)

        context = autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp) \
            if self.use_amp else nullcontext()
        with context:
            loss, _, _ = self.model(img)

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

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start = time.time()
        for batch_idx, img in enumerate(self.train_loader):
            loss = self._step(img, training=True)

            total_loss += loss
            if batch_idx % 10 == 0:
                print(f"    Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} ")
        tot_time = time.time() - start

                #       f"Loss: {loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.5f} | "
                #       f"Time: {metrics['time']:.2f}s")

        avg_loss = total_loss/ len(self.train_loader)
        self.loss['train'].append(avg_loss)
        print(f"\n[Train Epoch {epoch}] Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.5f}, Epoch time: {tot_time:.2f}\n")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss=0
        start = time.time()
        for batch_idx, img in enumerate(self.val_loader):
            loss = self._step(img, training=False)

            total_loss += loss
        tot_time = time.time() - start

            # if batch_idx % 10 == 0:
            #     print(f"Epoch {epoch} | Validation Batch {batch_idx}/{len(self.val_loader)} | "
            #           f"Loss: {loss:.4f} | Time: {metrics['time']:.2f}s")

        avg_loss = total_loss/ len(self.val_loader)
        self.loss['val'].append(avg_loss)
        print(f"\n[Validation Epoch {epoch}] Loss: {avg_loss:.4f}, Epoch time: {tot_time:.2f}\n")

    def save_checkpoint(self, path="best_model.pt"):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def save_encoder_checkpoint(self):
        checkpoint = {
            'encoder': self.model.encoder.state_dict()
        }
        torch.save(checkpoint, self.path)

    def train(self):
        best_loss = float('inf')
        for epoch in range(1, self.config["num_epochs"] + 1):
            self.train_epoch(epoch)
            self.validate(epoch)
            val_loss = self.loss['val'][epoch-1]
            if val_loss < best_loss:
                best_loss = val_loss
                # self.save_checkpoint()
                self.save_encoder_checkpoint()
                print(f"\nNew best model saved! Loss: {best_loss:.4f}\n")
        # torch.save(self.val_metrics,'validation_metrics_mae.pt')
        
        torch.save(self.loss, os.path.join(self.output_dir, f'{self.config["specific_name"]}_loss_segmentation.pt'))


if __name__ == "__main__":
    print("\nTesting MAETrainer with ViT MAE and ConvNeXtV2 MAE:")
    from torch.utils.data import DataLoader, TensorDataset
    from models.torch.vit import create_vit_mae
    from models.torch.convnextv2rms import create_convnextv2_mae

    # Dummy dataset: 10 samples, 12 channels, 256x256
    dummy_data = torch.randn(10, 12, 256, 256)
    dummy_loader = DataLoader(TensorDataset(dummy_data), batch_size=2)

    # Minimal config
    config = {
        'specific_name': 'test',
        'lr': 1e-4,
        'num_epochs': 1,
        'compile': False
    }
    device = torch.device('cpu')

    # Test ViT MAE with MAETrainer
    print("\n--- ViT MAE Trainer ---")
    vit_mae = create_vit_mae(size='base', in_channels=12, patch_size=16, mask_ratio=0.75)
    trainer_vit = MAETrainer(vit_mae, dummy_loader, dummy_loader, device, config)
    trainer_vit.train_epoch(1)
    trainer_vit.validate(1)

    # Test ConvNeXtV2 MAE with MAETrainer
    print("\n--- ConvNeXtV2 MAE Trainer ---")
    convnext_mae = create_convnextv2_mae(size='base', in_chans=12, mask_ratio=0.75)
    trainer_convnext = MAETrainer(convnext_mae, dummy_loader, dummy_loader, device, config)
    trainer_convnext.train_epoch(1)
    trainer_convnext.validate(1)
    
    # Example test logic for ViT MAE and ConvNeXtV2 MAE
    print("\nTesting ViT MAE and ConvNeXtV2 MAE forward pass:")
    from models.torch.vit import create_vit_mae
    from models.torch.convnextv2rms import create_convnextv2_mae

    # Dummy input
    x = torch.randn(2, 12, 256, 256)

    # Test ViT MAE
    vit_mae = create_vit_mae(size='base', in_channels=12, patch_size=16, mask_ratio=0.75)
    vit_loss, vit_pred, vit_mask = vit_mae(x)
    print(f"ViT MAE: loss={vit_loss.item():.4f}, pred shape={vit_pred.shape}, mask shape={vit_mask.shape}")

    # Test ConvNeXtV2 MAE
    convnext_mae = create_convnextv2_mae(size='base', in_chans=12, mask_ratio=0.75)
    convnext_loss, convnext_pred, convnext_mask = convnext_mae(x)
    print(f"ConvNeXtV2 MAE: loss={convnext_loss.item():.4f}, pred shape={convnext_pred.shape}, mask shape={convnext_mask.shape}")
    