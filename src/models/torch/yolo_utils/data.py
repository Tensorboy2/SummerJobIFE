import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
import numpy as np

class PTMultiTaskDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(os.path.join(self.folder, self.files[idx]))

        # Extract image and mask
        img = sample["image"].permute(2, 0, 1)/15  # [H, W, C] → [C, H, W]
        mask = sample["mask"]                  # [1, H, W]

        # Compute bounding box from mask
        y_indices, x_indices = torch.where(mask[0] > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            bbox = torch.tensor([0.5, 0.5, 0.0, 0.0])  # no object
            label = torch.tensor(0.0)
        else:
            H, W = mask.shape[-2:]
            x1, x2 = x_indices.min().float(), x_indices.max().float()
            y1, y2 = y_indices.min().float(), y_indices.max().float()

            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            bbox = torch.tensor([cx, cy, w, h])
            label = torch.tensor(1.0)

        if self.transform:
            img = self.transform(img)
        

        return img, mask, bbox, label

def get_dataloaders(config):
    # transform = T.Compose([
    #     T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     # T.ToDtype(torch.float32, scale=True),
    # ])

    data_path = 'src/data/processed'  # where the .pt files are
    full_dataset = PTMultiTaskDataset(data_path, transform=None)

    train_size = int((1 - config['val_ratio']) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = {
        'val_ratio': 0.2,
        'batch_size': 1
    }

    train, val = get_dataloaders(config=config)

    # Take one sample from the dataloader
    for img, mask, bbox, label in train:
        if label[0]==1:
            img = img[0]       # [C, H, W]
            mask = mask[0]     # [1, H, W]
            bbox = bbox[0]     # [4]
            label = label[0]   # []

            # Select Sentinel-2 RGB (bands 4, 3, 2) - assuming original 12-band image
            if img.shape[0] >= 4:
                rgb = img[[3, 2, 1]]#/20  # band indices for RGB (S2: B4, B3, B2)
            else:
                rgb = img[[0, 1, 2]]  # fallback for non-S2 or RGB-only

            rgb = rgb.permute(1, 2, 0).clip(0, 1).cpu().numpy()  # [H, W, 3]

            # Bounding box (denormalized)
            H, W = rgb.shape[:2]
            cx, cy, w, h = (bbox * torch.tensor([W, H, W, H])).tolist()
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)

            # Plot
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(rgb)
            axs[0].set_title("Sentinel-2 RGB")
            axs[0].add_patch(plt.Rectangle((x1, y1), w, h, edgecolor='lime', facecolor='none', linewidth=2))
            axs[0].text(x1, y1 - 5, f"Label: {int(label.item())}", color="white", backgroundcolor="black")

            axs[1].imshow(mask.squeeze().cpu(), cmap='gray')
            axs[1].set_title("Segmentation Mask")
            plt.tight_layout()
            plt.show()
            break  # remove if you want more examples
