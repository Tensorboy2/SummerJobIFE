import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
import os

class CustomDataset(Dataset):
    def __init__(self, folder, task="segmentation", transform=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(os.path.join(self.folder, self.files[idx]))
        x = item["image"].permute(2, 0, 1)  # HWC -> CHW
        if self.task == "segmentation":
            y = item["mask"]
        else:
            y = item["label"]

        # Convert to Image/Mask for v2 transforms
        if self.task == "segmentation":
            x = tv_tensors.Image(x)
            y = tv_tensors.Mask(y)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

def get_dataloaders(config):
    """get dataloaders"""

    transform = T.Compose([
        T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToDtype(torch.float32, scale=True),
    ])

    data_path = 'src/data/processed'
    full_dataset = CustomDataset(data_path, task='segmentation')
    num_samples = len(full_dataset)

    print(f"Num datapoints: {num_samples}")
    train_size = int((1 - config['val_ratio']) * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Now set transforms on the training subset only
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = None  # no augmentation for val

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == '__main__':
    config = {
        'val_ratio': 0.2,
        'batch_size': 32,
    }
    train, val = get_dataloaders(config=config)
    print(len(train))
    print(len(val))
