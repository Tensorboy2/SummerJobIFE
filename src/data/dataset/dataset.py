import torch
from torch.utils.data import Dataset, DataLoader
import os

# class transform: 


class CustomDataset(Dataset):
    def __init__(self, folder, task="segmentation"):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
        self.task = task

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(os.path.join(self.folder, self.files[idx]))
        x = item["image"].permute(2, 0, 1)  # HWC -> CHW
        y = item["mask"] if self.task == "segmentation" else item["label"]
        return x, y


def get_dataloaders(config):
    """get dataloaders"""
    data_path = 'src/data/processed'
    dataset = CustomDataset(data_path, task='segmentation')
    num_samples = len(dataset)

    print(f"Num datapoints: {num_samples}")
    train_size = int((1 - config['val_ratio']) * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=2,  
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
        drop_last=True  # Drop last incomplete batch for training
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=2,  
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        shuffle=False,
        drop_last=False  # Drop last incomplete batch for training
    )

    return train_data_loader, val_data_loader



if __name__ == '__main__':
    # data_path = 'src/data/processed'
    # dataset = dataset(data_path, task='segmentation')
    # dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=32,num_workers=2,pin_memory=False,persistent_workers=False)
    config = {
        'val_ratio':0.2,
        'batch_size':32,
    }
    train, val = get_dataloaders(config=config)
    print(len(train))
    print(len(val))