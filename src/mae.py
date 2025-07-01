from models.torch.convnextv2 import ConvNeXtV2MAE
from models.torch.yolo_utils.data import get_dataloaders
import torch.optim as optim
import matplotlib.pyplot as plt
from train_mae import MAETrainer
import torch
def train(trainer,config):
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    num_epochs = config['num_epochs']
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = trainer.train_epoch(epoch)
        
        # Validate
        val_metrics = trainer.validate(epoch)

def mae():
    config = {
        "val_ratio":0.2,
        "batch_size":32,
        'lr': 1e-2,
        'num_epochs': 100,
        'warmup_steps': 1000,
        'decay': 'cosine',
        'weight_decay': 0.1,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_loader, val_data_loaders = get_dataloaders(config)
    model = ConvNeXtV2MAE(in_chans=12).to(device=device)
    trainer = MAETrainer(model=model,
                         train_loader=train_data_loader, 
                         val_loader=val_data_loaders,
                         device=device,
                         config=config)
    train(trainer=trainer,config=config)

if __name__ == "__main__":
    mae()
    