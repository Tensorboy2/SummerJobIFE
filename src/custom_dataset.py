import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
import numpy as np

def get_mae_transforms():
    return T.Compose([
        # Optional: add noise or jitter for slight regularization
        T.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),  # Light Gaussian noise
        T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),        # Clamp after noise
        T.RandomHorizontalFlip(p=0.1),
        T.RandomVerticalFlip(p=0.1),
        # Optionally resize (depends on your ConvNeXt patch size)
        # T.Resize((256, 256)),  # Only if your data isn't already this size
    ])

class MaskedAutoEncoderDataset(Dataset):
    '''
    Dataset for training masked auto encoder.
    '''
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder+'/images') if f.endswith(".pt")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = torch.load(os.path.join(self.folder, 'images', self.files[idx]))
        # Extract image and mask
        img = image.permute(2, 0, 1)/15  # [H, W, C] → [C, H, W]

        if self.transform:
            img = self.transform(img)
    
        return img
    
class SegmentationDataset(Dataset):
    '''
    Dataset for training on segmentation.
    '''
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder+'/images') if f.endswith(".pt")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = torch.load(os.path.join(self.folder, 'images', self.files[idx]))
        mask = torch.load(os.path.join(self.folder, 'masks', self.files[idx]))

        # Extract image and mask
        img = image.permute(2, 0, 1)/15  # [H, W, C] → [C, H, W]
        mask = mask                 # [1, H, W]

        if self.transform:
            img = self.transform(img)
    
        return img, mask

def get_dataloaders(config):
    '''
    Function for fetching DataLoaders for training and validation.
    '''
    data_path = 'src/data/processed_unique'  # where the .pt files are


    if config['data_type']=='mae':
    
        full_dataset = MaskedAutoEncoderDataset(data_path, transform=get_mae_transforms())
    
    elif config['data_type']=='segmentation':

        full_dataset = SegmentationDataset(data_path, transform=None)


    train_size = int((1 - config['val_ratio']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )


    print(
        f'Dataset:\n'
        f'-  Type: {config["data_type"]}\n'
        f'-  Length: {len(full_dataset)}\n'
        f'-  Validation ratio: {config["val_ratio"]}\n'
        f'-  Batch size: {config["batch_size"]}\n'
        )

    return train_loader, val_loader

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = {
        'val_ratio': 0.2,
        'batch_size': 2,
        'data_type':'segmentation'
    }

    train, val = get_dataloaders(config=config)

    for img, mask in train:
        print(img.shape)
        y_indices, x_indices = torch.where(mask[0,0] > 0)
        if len(x_indices) > 0 or len(y_indices) > 0:
            img = img[0]       # [C, H, W]
            mask = mask[0]     # [1, H, W]

            # Select Sentinel-2 RGB (bands 4, 3, 2) - assuming original 12-band image
            if img.shape[0] >= 4:
                rgb = img[[3, 2, 1]]#/20  # band indices for RGB (S2: B4, B3, B2)
            else:
                rgb = img[[0, 1, 2]]  # fallback for non-S2 or RGB-only

            rgb = rgb.permute(1, 2, 0).clip(0, 1).cpu().numpy()  # [H, W, 3]

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(rgb)
            axs[0].set_title("Sentinel-2 RGB")

            axs[1].imshow(mask.squeeze().cpu(), cmap='gray')
            axs[1].set_title("Segmentation Mask")
            plt.tight_layout()
            plt.show()
            break  # remove if you want more examples
