import torch
import torch.optim as to

from trainer import EfficientTrainer
from models.torch.u_net import UNetWithSkips, MiniUNet
from data.dataset.dataset import get_dataloaders

config = {
        'val_ratio':0.2,
        'batch_size':64,
        'use_amp':True,
        'compile':True,
        'lr':1e-4,
        'weight_decay':0.05,
        'num_epochs':2,
        'warmup_steps':10,
        'decay': 'cosine',
    }

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MiniUNet().to(device=device)
    # model = UNetWithSkips().to(device=device)
    # print(model)
    train, val = get_dataloaders(config)
    trainer = EfficientTrainer(model=model,
                               train_loader=train,
                               val_loader=val,
                               device=device,
                               config=config)
    # print(config)
    for i in range(config['num_epochs']):
        trainer.train_epoch(i)
        trainer.validate()

if __name__ == '__main__':
    main(config)