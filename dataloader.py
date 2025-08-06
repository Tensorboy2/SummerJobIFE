from torch.utils.data import Dataset, DataLoader
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, training=True, crop_size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
        self.training = training
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = torch.load(img_path).permute(2, 0, 1).float()
        mask = torch.load(mask_path).float()


        mask_min, mask_max = mask.min(), mask.max()
        if mask_max > mask_min:
            mask = (mask - mask_min) / (mask_max - mask_min)
        else:
            mask = torch.zeros_like(mask)


        # Apply augmentations if training
        if self.training:
            img, mask = self.random_flip(img, mask)
            img, mask = self.random_rotate(img, mask)
            img = self.random_noise(img)
            img = self.random_brightness(img)
            img = self.random_channel_dropout(img)
            img, mask = self.random_crop(img, mask, self.crop_size)

        return img, mask

    # ----------- Augmentations -----------
    def random_flip(self, img, mask):
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[1])  # vertical
            mask = torch.flip(mask, dims=[0])
        return img, mask

    def random_rotate(self, img, mask):
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])
        return img, mask

    def random_noise(self, img):
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img + noise, 0., 1.)
        return img

    def random_brightness(self, img):
        if torch.rand(1) < 0.3:
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
            img = torch.clamp(img * factor, 0., 1.)
        return img

    def random_channel_dropout(self, img):
        if torch.rand(1) < 0.2:
            c = torch.randint(0, img.shape[0], (1,)).item()
            img[c] = 0
        return img

    def random_crop(self, img, mask, crop_size=(128, 128)):
        H, W = img.shape[1], img.shape[2]
        ch, cw = crop_size
        if H < ch or W < cw:
            return img, mask  # skip crop
        top = torch.randint(0, H - ch + 1, (1,)).item()
        left = torch.randint(0, W - cw + 1, (1,)).item()
        img = img[:, top:top+ch, left:left+cw]
        mask = mask[top:top+ch, left:left+cw]
        return img, mask


def get_dataloaders(config):
    root_path = 'src/data/processed_unique'
    image_dir = os.path.join(root_path, 'images')
    mask_dir = os.path.join(root_path, 'masks')
    
    full_dataset = CustomDataset(image_dir, mask_dir, training=True, crop_size=config.get("crop_size", (128, 128)))
    print(f"Dataset size: {len(full_dataset)}")
    
    val_size = int(len(full_dataset) * config['val_ratio'])
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Set val_dataset to eval mode (no augmentations)
    val_dataset.dataset.training = False

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 0)
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 0)
    )

    return train_loader, val_loader