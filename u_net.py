import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from convnext_v3 import create_convnextv3_segmentation

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=12, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_ch, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        self.name = 'unet_2'
    #     self._init_weights()

    # def _init_weights(self):
    #     # Kaiming normal for conv layers, zero for classifier
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m is self.classifier:
    #                 nn.init.zeros_(m.weight)
    #                 if m.bias is not None:
    #                     nn.init.zeros_(m.bias)
    #             else:
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #                 if m.bias is not None:
    #                     nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
    


import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # [B, C, H, W] -> [B, H, W, C] for LayerNorm/Linear
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class Decoder(nn.Module):
    def __init__(self, encoder_output_channels, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(encoder_output_channels, 256, kernel_size=1)
        self.block1 = ConvNeXtBlock(256)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.block2 = ConvNeXtBlock(128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.block3 = ConvNeXtBlock(64)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.block4 = ConvNeXtBlock(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming normal for conv layers, zero for classifier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.conv4(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

    
class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.initial_conv = full_model.initial_conv
        self.stem = full_model.stem
        self.downsample_layers = full_model.downsample_layers
        self.stages = full_model.stages
        # self.norm = LayerNorm2d()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stem(x)

        x = self.stages[0](x)
        x = self.downsample_layers[0](x)

        x = self.stages[1](x)
        x = self.downsample_layers[1](x)

        x = self.stages[2](x)
        x = self.downsample_layers[2](x)

        x = self.stages[3](x)
        # x = self.norm(x)
        return x

    
class ConvNeXtV2Segmentation(nn.Module):
    def __init__(self, in_chans=12, num_classes=1, encoder_output_channels=320):
        super().__init__()
        mmearth_model = torch.hub.load('vishalned/mmearth-train', 'MPMAE', model_name='convnextv2_atto', pretrained=True, linear_probe=True,verbose=True)
        encoder = ConvNeXtV2Encoder(mmearth_model)
        self.encoder = encoder
        self.decoder = Decoder(encoder_output_channels=encoder_output_channels)
        self.num_classes = num_classes
        self.name = 'convnextv2_open'
        # self.sigmoid = nn.Sigmoid()
        # Freeze encoder weights
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def iou_score(preds, targets, threshold=0.5):
    """Calculate IoU score for batch of predictions and targets"""
    # Apply sigmoid to predictions if they're logits
    if preds.max() > 1 or preds.min() < 0:
        preds = torch.sigmoid(preds)
    
    preds = (preds > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    
    # Avoid division by zero
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean()

def dice_loss(preds, targets, smooth=1e-8):
    """Dice loss for better segmentation training"""
    preds = torch.sigmoid(preds)
    
    intersection = (preds * targets).sum(dim=(2, 3))
    dice_coeff = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return 1 - dice_coeff.mean()

def focal_loss(preds, targets, alpha=1, gamma=2, smooth=1e-8):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for rare class (default: 1)
        gamma: Focusing parameter (default: 2)
    """
    # Apply sigmoid to get probabilities
    prob = torch.sigmoid(preds)
    
    # Calculate BCE loss
    bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
    
    # Calculate p_t
    p_t = prob * targets + (1 - prob) * (1 - targets)
    
    # Calculate alpha_t
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Calculate focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss = focal_weight * bce_loss
    
    return focal_loss.mean()

def combined_loss(preds, targets, loss_weights=None):
    """
    Combined loss function with multiple options
    
    Args:
        loss_weights: Dict with keys 'bce', 'dice', 'focal' and their respective weights
    """
    if loss_weights is None:
        loss_weights = {'bce': 0.5, 'dice': 0.5, 'focal': 0.0}
    
    total_loss = 0.0
    
    if loss_weights.get('bce', 0) > 0:
        bce = nn.BCELoss()(preds, targets)
        total_loss += loss_weights['bce'] * bce
    
    if loss_weights.get('dice', 0) > 0:
        dice = dice_loss(preds, targets)
        total_loss += loss_weights['dice'] * dice
    
    if loss_weights.get('focal', 0) > 0:
        focal = focal_loss(preds, targets, alpha=0.75, gamma=2)
        total_loss += loss_weights['focal'] * focal
    
    return total_loss

# --- Metric helpers ---
def bce_loss(preds, targets):
    return nn.BCEWithLogitsLoss()(preds, targets)

def dice_coeff(preds, targets, smooth=1e-8):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return dice.mean()

def get_confusion_matrix(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    return tp, fp, tn, fn

def precision_score(tp, fp):
    return tp / (tp + fp + 1e-8)

def recall_score(tp, fn):
    return tp / (tp + fn + 1e-8)

def f1_score(tp, fp, fn):
    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)
    return 2 * (prec * rec) / (prec + rec + 1e-8)

def plot_example(img, mask, pred, epoch=None, batch=None):
    """Plot example with proper handling of different input formats"""
    plt.figure(figsize=(12, 5))
    
    # Handle image display - assume first 3 channels for RGB visualization
    plt.subplot(1, 3, 1)
    if img.shape[0] >= 3:  # If we have at least 3 channels
        # Take first 3 channels and normalize for display
        img_display = img[[3,2,1]].permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        plt.imshow(img_display)
    else:
        plt.imshow(img[0].cpu().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    pred_binary = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy()
    plt.imshow(pred_binary, cmap='gray')
    title = 'Prediction'
    if epoch is not None and batch is not None:
        title += f' (E{epoch}, B{batch})'
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# class CustomDataset(Dataset):
#     def __init__(self, image_dir, mask_dir):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])
#         self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])
#         assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
#         img = torch.load(img_path).permute(2, 0, 1).float()
#         img_min, img_max = img.min(), img.max()
#         if img_max > img_min:
#             img = (img - img_min) / (img_max - img_min)
#         else:
#             img = torch.zeros_like(img)

#         mask = torch.load(mask_path).float()
#         mask_min, mask_max = mask.min(), mask.max()
#         if mask_max > mask_min:
#             mask = (mask - mask_min) / (mask_max - mask_min)
#         else:
#             mask = torch.zeros_like(mask)

#         return img, mask

# def get_dataloaders(config):
#     root_path = 'src/data/processed_unique'
#     image_dir = os.path.join(root_path, 'images')
#     mask_dir = os.path.join(root_path, 'masks')
    
#     dataset = CustomDataset(image_dir, mask_dir)
#     print(f"Dataset size: {len(dataset)}")
    
#     # Split dataset
#     val_size = int(len(dataset) * config['val_ratio'])
#     train_size = len(dataset) - val_size
    
#     # Use fixed seed for reproducible splits
#     generator = torch.Generator().manual_seed(42)
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size], generator=generator
#     )
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=config['batch_size'], 
#         shuffle=True, 
#         num_workers=config.get('num_workers', 0)
#     )
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=config['batch_size'], 
#         shuffle=False, 
#         num_workers=config.get('num_workers', 0)
#     )
    
#     return train_loader, val_loader

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

        # Normalize image
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min + 1e-5)
        else:
            img = torch.zeros_like(img)

        # mask_min, mask_max = mask.min(), mask.max()
        # if mask_max > mask_min:
        #     mask = (mask - mask_min) / (mask_max - mask_min + 1e-5)
        # else:
        #     mask = torch.zeros_like(mask)

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

def train_model():
    # Configuration
    config = {
        'batch_size': 64,
        'val_ratio': 0.2,
        'num_workers': 4,
        'learning_rate': 1e-3,  # Lower learning rate
        'num_epochs': 40,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Loss function weights - experiment with these!
        'loss_weights': {
            'bce': 1.0,     # Standard BCE
            'dice': 0.0,    # Dice loss for overlap
            'focal': 0.0,    # Focal loss for hard examples
        },
        'crop_size': (128, 128),  # Crop size for training augmentations
        'weight_decay': 0.1,  # Regularization
        'warmup_steps': 00,  # No warmup for simplicity
        'learning_rate_decay': 'cosine',  # Use learning rate decay
        'plot_examples': False,  # Whether to plot examples during training
        'save_best_model': True  # Whether to save the best model based on validation Io

    }
    
    # Initialize model
    # model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1, encoder_output_channels=320)
    model = UNet(in_ch=12, out_ch=1)
    # model = create_convnextv3_segmentation(in_chans=12, num_classes=1, size='atto')
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Loss and optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
    # Warmup scheduler: linearly increase LR for a few epochs, then use ReduceLROnPlateau

    def lr_lambda(step):
        if step < config.get('warmup_steps', 0):
            return float(step + 1) / float(config.get('warmup_steps', 1e-8)) # Linear warmup
        elif step > config.get('warmup_steps', 0) and config['learning_rate_decay'] == 'cosine':
            # Cosine decay after warmup:
            return 0.5 * (1 + np.cos(np.pi * (step - config.get('warmup_steps', 0)) / (config['num_epochs'] * len(train_loader) - config.get('warmup_steps', 0))))
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    train_losses = []
    train_ious = []
    train_bces = []
    train_dices = []
    train_tps = []
    train_fps = []
    train_tns = []
    train_fns = []
    train_f1s = []
    train_recalls = []
    train_precisions = []

    val_losses = []
    val_ious = []
    val_bces = []
    val_dices = []
    val_tps = []
    val_fps = []
    val_tns = []
    val_fns = []
    val_f1s = []
    val_recalls = []
    val_precisions = []

    best_val_iou = 0.0
    
    print(f"Training on device: {device}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    for epoch in range(config['num_epochs']):
        print('\nTraining...')
        model.train()
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        epoch_train_bce = 0.0
        epoch_train_dice = 0.0
        epoch_train_tp = 0
        epoch_train_fp = 0
        epoch_train_tn = 0
        epoch_train_fn = 0

        for train_batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, masks, config['loss_weights'])

            # Calculate metrics
            with torch.no_grad():
                iou = iou_score(outputs.float(), masks.float())
                bce = bce_loss(outputs, masks)
                dice = dice_coeff(outputs, masks)
                tp, fp, tn, fn = get_confusion_matrix(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            epoch_train_loss += loss.item()
            epoch_train_iou += iou.item()
            epoch_train_bce += bce.item()
            epoch_train_dice += dice.item()
            epoch_train_tp += tp
            epoch_train_fp += fp
            epoch_train_tn += tn
            epoch_train_fn += fn

            if train_batch_idx % 10 == 0:
                avg_loss = epoch_train_loss / (train_batch_idx + 1)
                avg_iou = epoch_train_iou / (train_batch_idx + 1)
                avg_bce = epoch_train_bce / (train_batch_idx + 1)
                avg_dice = epoch_train_dice / (train_batch_idx + 1)
                print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {train_batch_idx}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.8f}, BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f}")

        # Calculate average training metrics
        n_train_batches = train_batch_idx + 1
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_train_iou = epoch_train_iou / n_train_batches
        avg_train_bce = epoch_train_bce / n_train_batches
        avg_train_dice = epoch_train_dice / n_train_batches
        avg_train_tp = epoch_train_tp
        avg_train_fp = epoch_train_fp
        avg_train_tn = epoch_train_tn
        avg_train_fn = epoch_train_fn
        avg_train_precision = precision_score(avg_train_tp, avg_train_fp)
        avg_train_recall = recall_score(avg_train_tp, avg_train_fn)
        avg_train_f1 = f1_score(avg_train_tp, avg_train_fp, avg_train_fn)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_iou = 0.0
        epoch_val_bce = 0.0
        epoch_val_dice = 0.0
        epoch_val_tp = 0
        epoch_val_fp = 0
        epoch_val_tn = 0
        epoch_val_fn = 0
        print("\nValidating...")
        with torch.no_grad():
            for val_batch_idx , (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = combined_loss(outputs, masks, config['loss_weights'])
                val_iou = iou_score(outputs, masks)
                bce = bce_loss(outputs, masks)
                dice = dice_coeff(outputs, masks)
                tp, fp, tn, fn = get_confusion_matrix(outputs, masks)

                epoch_val_loss += val_loss.item()
                epoch_val_iou += val_iou.item()
                epoch_val_bce += bce.item()
                epoch_val_dice += dice.item()
                epoch_val_tp += tp
                epoch_val_fp += fp
                epoch_val_tn += tn
                epoch_val_fn += fn

                if val_batch_idx % 10 == 0:
                    avg_loss = epoch_val_loss / (val_batch_idx + 1)
                    avg_iou = epoch_val_iou / (val_batch_idx + 1)
                    avg_bce = epoch_val_bce / (val_batch_idx + 1)
                    avg_dice = epoch_val_dice / (val_batch_idx + 1)
                    print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                        f"Batch {val_batch_idx}/{len(val_loader)}, "
                        f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.8f}, BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f}")

        n_val_batches = val_batch_idx + 1
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_val_iou = epoch_val_iou / n_val_batches
        avg_val_bce = epoch_val_bce / n_val_batches
        avg_val_dice = epoch_val_dice / n_val_batches
        avg_val_tp = epoch_val_tp
        avg_val_fp = epoch_val_fp
        avg_val_tn = epoch_val_tn
        avg_val_fn = epoch_val_fn
        avg_val_precision = precision_score(avg_val_tp, avg_val_fp)
        avg_val_recall = recall_score(avg_val_tp, avg_val_fn)
        avg_val_f1 = f1_score(avg_val_tp, avg_val_fp, avg_val_fn)

        # Store history
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_bces.append(avg_train_bce)
        train_dices.append(avg_train_dice)
        train_tps.append(avg_train_tp)
        train_fps.append(avg_train_fp)
        train_tns.append(avg_train_tn)
        train_fns.append(avg_train_fn)
        train_f1s.append(avg_train_f1)
        train_recalls.append(avg_train_recall)
        train_precisions.append(avg_train_precision)

        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_bces.append(avg_val_bce)
        val_dices.append(avg_val_dice)
        val_tps.append(avg_val_tp)
        val_fps.append(avg_val_fp)
        val_tns.append(avg_val_tn)
        val_fns.append(avg_val_fn)
        val_f1s.append(avg_val_f1)
        val_recalls.append(avg_val_recall)
        val_precisions.append(avg_val_precision)

        # Save best model
        if avg_val_iou > best_val_iou and config.get('save_best_model', False):
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f'{model.name}_'+'best_unet_model.pth')
            print(f"New best model saved with validation IoU: {best_val_iou:.4f}")

        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train BCE: {avg_train_bce:.4f}, Train Dice: {avg_train_dice:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val BCE: {avg_val_bce:.4f}, Val Dice: {avg_val_dice:.4f}")
        print(f"Train TP: {avg_train_tp}, FP: {avg_train_fp}, TN: {avg_train_tn}, FN: {avg_train_fn}, F1: {avg_train_f1:.4f}, Recall: {avg_train_recall:.4f}, Precision: {avg_train_precision:.4f}")
        print(f"Val   TP: {avg_val_tp}, FP: {avg_val_fp}, TN: {avg_val_tn}, FN: {avg_val_fn}, F1: {avg_val_f1:.4f}, Recall: {avg_val_recall:.4f}, Precision: {avg_val_precision:.4f}")
        print("-" * 80)

    return model, train_losses, train_ious, val_losses, val_ious, train_bces, val_bces, train_dices, val_dices, train_tps, val_tps, train_fps, val_fps, train_tns, val_tns, train_fns, val_fns, train_f1s, val_f1s, train_recalls, val_recalls, train_precisions, val_precisions

if __name__ == "__main__":
    import pandas as pd
    (
        model, train_losses, train_ious, val_losses, val_ious,
        train_bces, val_bces, train_dices, val_dices,
        train_tps, val_tps, train_fps, val_fps, train_tns, val_tns, train_fns, val_fns,
        train_f1s, val_f1s, train_recalls, val_recalls, train_precisions, val_precisions
    ) = train_model()

    # Save metrics as a pandas DataFrame
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_ious,
        'val_iou': val_ious,
        'train_bce': train_bces,
        'val_bce': val_bces,
        'train_dice': train_dices,
        'val_dice': val_dices,
        'train_tp': train_tps,
        'val_tp': val_tps,
        'train_fp': train_fps,
        'val_fp': val_fps,
        'train_tn': train_tns,
        'val_tn': val_tns,
        'train_fn': train_fns,
        'val_fn': val_fns,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'train_recall': train_recalls,
        'val_recall': val_recalls,
        'train_precision': train_precisions,
        'val_precision': val_precisions
    })
    metrics_df.to_csv(f"{model.name}_metrics.csv", index=False)
    print(f"Metrics saved to {model.name}_metrics.csv")