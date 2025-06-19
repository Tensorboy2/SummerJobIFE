import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import math
from contextlib import nullcontext

class EfficientTrainer:
    """Production-ready trainer with all optimizations"""
    
    def __init__(self, model, train_loader, val_loader, device, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # if config ==None:
        #     self.config = config
        # else:
        #     config={
        #         'use_amp':True,
        #         'compile':True,
        #         'lr':1e-4,
        #         'weight_decay':0.05,
        #     }
        
        # Mixed precision setup
        self.scaler = GradScaler()
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
        num_batches = len(self.train_loader)
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), \
                          target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision context
            with autocast(self.device) if self.use_amp else nullcontext():
                output = self.model(data)
                # loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
                loss = F.binary_cross_entropy_with_logits(output, target)
            
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
            
            # Logging every N steps
            # if batch_idx % 1 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}')
                
            self.scheduler.step()
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for data, target in self.val_loader:
            data, target = data.to(self.device, non_blocking=True), \
                          target.to(self.device, non_blocking=True)
            
            with autocast() if self.use_amp else nullcontext():
                output = self.model(data)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
                
            total_loss += loss.item()
            
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, loss, filepath):
        """Efficient checkpointing"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Async saving to avoid blocking training
        torch.save(checkpoint, filepath)

# if __name__ == '__main__':
#     trainer = EfficientTrainer()