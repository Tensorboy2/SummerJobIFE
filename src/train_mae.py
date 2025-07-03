import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from collections import defaultdict
import math
import time


class MAETrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.path=f"pretrained_encoder_{self.config['specific_name']}.pt"

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
        img = img.to(self.device, non_blocking=True)

        context = autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp) \
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
        torch.save(self.loss,'loss_mae.pt')
