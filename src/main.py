import torch
# import torch.optim as to
import os

from trainer import EfficientTrainer
from models.torch.u_net import UNet, HalfUNet
from models.torch.YOLOv8 import YOLOMultiTask
from models.torch.yolo_utils.data import get_dataloaders

def train_model(trainer, num_epochs, checkpoint_dir, save_every=5, plot_every=1):
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
        
        # Save checkpoint for best model (using dice as primary metric)
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            trainer.save_checkpoint(epoch, val_metrics, checkpoint_dir)
            print(f"New best Dice score: {best_dice:.4f}")
        
        # Regular checkpointing
        if epoch % save_every == 0:
            trainer.save_checkpoint(epoch, val_metrics, checkpoint_dir)
        
        # Plot metrics
        if epoch % plot_every == 0:
            plot_path = os.path.join(checkpoint_dir, f'metrics_epoch_{epoch}.png')
            trainer.plot_metrics(save_path=plot_path)
    
    # Final summary
    best_metrics = trainer.get_best_metrics()
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE - BEST METRICS:")
    print(f"{'='*50}")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

config = {
        'lr': 1e-2,
        'num_epochs': 100,
        'warmup_steps': 1000,
        'decay': 'cosine',
        'weight_decay': 0.1,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
        'bce_weight': 0.4,    
        'dice_weight': 0.6,   
        'val_ratio': 0.2,
        'batch_size':256
    }

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNet(in_ch=12).to(device=device)
    model = YOLOMultiTask(in_ch=12,input_size=256).to(device=device)
    train, val = get_dataloaders(config)
    trainer = EfficientTrainer(model=model,
                               train_loader=train,
                               val_loader=val,
                               device=device,
                               config=config)
    
    train_model(trainer=trainer,
                num_epochs=config['num_epochs'],
                checkpoint_dir='src/checkpoints')

if __name__ == '__main__':
    main(config)