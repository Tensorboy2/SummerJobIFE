from models.torch.convnextv2 import ConvNeXtV2MAE
from models.torch.yolo_utils.data import get_dataloaders
import torch.optim as optim
import matplotlib.pyplot as plt
from train_mae import MAETrainer

def train(trainer,config):
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_dice = 0.0
    
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
        "batch_size":32
    }

    train_data_loader, val_data_loaders = get_dataloaders(config)
    model = ConvNeXtV2MAE(in_chans=12)
    trainer = MAETrainer(model=model, train_data_loader=train_data_loader, val_data_loader=val_data_loaders)
    train(trainer=trainer)

if __name__ == "__main__":
    mae()
    