import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from collections import defaultdict
import math
import time


class MAETrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

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
        self.train_history = {'loss': []}
        self.val_history = {'loss': []}

        if hasattr(torch, 'compile') and config.get('compile', True):
            self.model = torch.compile(self.model)

    def _build_scheduler(self):
        total_steps = self.config['num_epochs'] * len(self.train_loader)
        warmup_steps = self.config.get('warmup_steps', 100)
        decay_type = self.config.get('decay', 'linear')

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            if decay_type == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            elif decay_type == "linear":
                return max(0.0, 1.0 - progress)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _step(self, img, mask, bbox, label, training=True):
        img, mask, bbox, label = map(
            lambda x: x.to(self.device, non_blocking=True),
            (img, mask, bbox, label)
        )

        context = autocast(device_type=self.device, dtype=torch.bfloat16) if self.use_amp else nullcontext()
        with context:
            loss, pred, out_mask = self.model(img)

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
        metrics = defaultdict(float)
        for batch_idx, (img, mask, bbox, label) in enumerate(self.train_loader):
            start = time.time()
            loss = self._step(img, mask, bbox, label, training=True)
            metrics['loss'] += loss
            metrics['time'] += time.time() - start

            if batch_idx % 1 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.5f} | "
                      f"Time: {metrics['time']:.2f}s")

        metrics = {k: v / len(self.train_loader) for k, v in metrics.items()}
        self.train_history['loss'].append(metrics['loss'])

        print(f"\n[Train Epoch {epoch}] Loss: {metrics['loss']:.4f}")
        return metrics

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        metrics = defaultdict(float)
        for batch_idx, (img, mask, bbox, label) in enumerate(self.val_loader):
            start = time.time()
            loss = self._step(img, mask, bbox, label, training=False)
            metrics['loss'] += loss
            metrics['time'] += time.time() - start

            if batch_idx % 1 == 0:
                print(f"Epoch {epoch} | Validation Batch {batch_idx}/{len(self.val_loader)} | "
                      f"Loss: {loss:.4f} | Time: {metrics['time']:.2f}s")

        metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        self.val_history['loss'].append(metrics['loss'])

        print(f"\n[Validation Epoch {epoch}] Loss: {metrics['loss']:.4f}")
        return metrics
