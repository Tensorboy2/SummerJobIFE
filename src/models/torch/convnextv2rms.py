import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):
    '''
    Global Response Normalization layer.
    '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=(2, 3), keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class RMSNorm(nn.Module):
    '''
    RMSNorm for token sequences ([B, ..., C] before Linear).
    '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.scale

class RMSNorm2d(nn.Module):
    '''
    RMSNorm for 2D convolutional features [B, C, H, W].
    '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.scale

class ConvNeXtBlock(nn.Module):
    '''

    ConvNeXt v2 block using GRN and RMSNorm.

    '''
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = RMSNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3,3,9,3], dims=[96,192,384,768]):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            RMSNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                RMSNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        for i, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = downsample(x)
            x = stage(x)
            features.append(x)
        return x, features

class ConvNeXtV2Decoder(nn.Module):
    def __init__(self, encoder_dims=[96,192,384,768], decoder_dims=[512,256,128,64], out_chans=12):
        super().__init__()
        self.encoder_to_decoder = nn.Conv2d(encoder_dims[-1], decoder_dims[0], kernel_size=1)

        self.upsample_layers = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i+1], kernel_size=2, stride=2),
                RMSNorm2d(decoder_dims[i+1])
            ))

        self.final_upsample = nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4)

        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_dims)):
            blocks = nn.Sequential(
                *[ConvNeXtBlock(dim=decoder_dims[i]) for _ in range(2)]
            )
            self.decoder_stages.append(blocks)

        self.output_proj = nn.Conv2d(decoder_dims[-1], out_chans, kernel_size=1)

    def forward(self, x):
        x = self.encoder_to_decoder(x)
        for i, (stage, upsample) in enumerate(zip(self.decoder_stages[:-1], self.upsample_layers)):
            x = stage(x)
            x = upsample(x)
        x = self.decoder_stages[-1](x)
        x = self.final_upsample(x)
        x = self.output_proj(x)
        return x

class ConvNeXtV2SegmentationDecoder(nn.Module):
    def __init__(self, encoder_dims=[96,192,384,768], decoder_dims=[384,192,96,48], num_classes=21):
        super().__init__()
        self.skip_projections = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.skip_projections.append(
                nn.Conv2d(encoder_dims[-(i+2)], decoder_dims[i+1], kernel_size=1)
            )

        self.encoder_to_decoder = nn.Conv2d(encoder_dims[-1], decoder_dims[0], kernel_size=1)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            fusion = nn.Sequential(
                nn.Conv2d(decoder_dims[i+1]*2, decoder_dims[i+1], kernel_size=3, padding=1),
                RMSNorm2d(decoder_dims[i+1]),
                nn.GELU()
            )
            blocks = nn.Sequential(*[ConvNeXtBlock(dim=decoder_dims[i+1]) for _ in range(2)])
            self.decoder_blocks.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i+1], kernel_size=2, stride=2),
                'fusion': fusion,
                'blocks': blocks
            }))

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            RMSNorm2d(decoder_dims[-1]),
            nn.GELU(),
            *[ConvNeXtBlock(dim=decoder_dims[-1]) for _ in range(2)]
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1], kernel_size=3, padding=1),
            RMSNorm2d(decoder_dims[-1]),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)
        )

    def forward(self, encoder_features):
        x = self.encoder_to_decoder(encoder_features[-1])
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block['upsample'](x)
            skip = self.skip_projections[i](encoder_features[-(i+2)])
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder_block['fusion'](x)
            x = decoder_block['blocks'](x)
        x = self.final_upsample(x)
        return self.classifier(x)

class ConvNeXtV2MAE(nn.Module):
    def __init__(self, in_chans=12, depths=[3,3,9,3], dims=[96,192,384,768],
                 decoder_dims=[512,256,128,64], mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = 16
        self.in_chans = in_chans
        self.encoder = ConvNeXtV2Encoder(in_chans, depths, dims)
        self.decoder = ConvNeXtV2Decoder(dims, decoder_dims, in_chans)
        self.mask_token = nn.Parameter(torch.zeros(1, dims[-1], 1, 1))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * self.in_chans)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], self.in_chans, h * p, h * p)

    def random_masking(self, x, mask_ratio):
        N, C, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).reshape(N, 1, H, W)
        return mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        mask, ids_restore = self.random_masking(x, mask_ratio)
        x_masked = x * (1 - mask)
        latent, features = self.encoder(x_masked)
        return latent, mask, ids_restore, features

    def forward_decoder(self, x, ids_restore=None):
        return self.decoder(x)

    def forward_loss(self, imgs, pred, mask):
        loss = (pred - imgs).pow(2).mean(dim=1, keepdim=True)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore, features = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class ConvNeXtV2Segmentation(nn.Module):
    def __init__(self, in_chans=3, num_classes=21, depths=[3,3,9,3], dims=[96,192,384,768],
                 decoder_dims=[384,192,96,48]):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(in_chans, depths, dims)
        self.decoder = ConvNeXtV2SegmentationDecoder(dims, decoder_dims, num_classes)

    def forward(self, x):
        _, encoder_features = self.encoder(x)
        return self.decoder(encoder_features)

class ConvNeXtV2(ConvNeXtV2Encoder):
    def forward(self, x):
        latent, features = super().forward(x)
        return latent

# --- Model creation functions for different sizes (like ViT) ---
def create_convnextv2_segmentation(size='base', in_chans=3, num_classes=1):
    """Create a ConvNeXtV2 model for segmentation tasks with different sizes."""
    configs = {
        'base': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768],
            'decoder_dims': [384, 192, 96, 48]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536],
            'decoder_dims': [768, 384, 192, 96]
        },
        'small': {
            'depths': [3, 3, 9, 3],
            'dims': [64, 128, 256, 512],
            'decoder_dims': [256, 128, 64, 32]
        },
        'nano': {
            'depths': [2, 2, 6, 2],
            'dims': [48, 96, 192, 384],
            'decoder_dims': [192, 96, 48, 24]
        },
        'pico': {
            'depths': [2, 2, 4, 2],
            'dims': [32, 64, 128, 256],
            'decoder_dims': [128, 64, 32, 16]
        },
        'femto': {
            'depths': [1, 2, 4, 2],
            'dims': [24, 48, 96, 192],
            'decoder_dims': [96, 48, 24, 12]
        },
        'atto': {
            'depths': [1, 2, 2, 2],
            'dims': [16, 32, 64, 128],
            'decoder_dims': [64, 32, 16, 8]
        }
    }
    if size not in configs:
        raise ValueError(f"Unsupported ConvNeXtV2 size: {size}. Choose from {list(configs.keys())}")
    cfg = configs[size]
    return ConvNeXtV2Segmentation(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=cfg['depths'],
        dims=cfg['dims'],
        decoder_dims=cfg['decoder_dims']
    )

def create_convnextv2_mae(size='base', in_chans=12, mask_ratio=0.75):
    """Create a ConvNeXtV2 model for MAE pretraining with different sizes."""
    configs = {
        'base': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768],
            'decoder_dims': [512, 256, 128, 64]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'dims': [192, 384, 768, 1536],
            'decoder_dims': [1024, 512, 256, 128]
        },
        'small': {
            'depths': [3, 3, 9, 3],
            'dims': [64, 128, 256, 512],
            'decoder_dims': [256, 128, 64, 32]
        },
        'nano': {
            'depths': [2, 2, 6, 2],
            'dims': [48, 96, 192, 384],
            'decoder_dims': [192, 96, 48, 24]
        },
        'pico': {
            'depths': [2, 2, 4, 2],
            'dims': [32, 64, 128, 256],
            'decoder_dims': [128, 64, 32, 16]
        },
        'femto': {
            'depths': [1, 2, 4, 2],
            'dims': [24, 48, 96, 192],
            'decoder_dims': [96, 48, 24, 12]
        },
        'atto': {
            'depths': [1, 2, 2, 2],
            'dims': [16, 32, 64, 128],
            'decoder_dims': [64, 32, 16, 8]
        }
    }
    if size not in configs:
        raise ValueError(f"Unsupported ConvNeXtV2 size: {size}. Choose from {list(configs.keys())}")
    cfg = configs[size]
    return ConvNeXtV2MAE(
        in_chans=in_chans,
        depths=cfg['depths'],
        dims=cfg['dims'],
        decoder_dims=cfg['decoder_dims'],
        mask_ratio=mask_ratio
    )

if __name__ == '__main__':
    # Test MAE model creation functions
    print("Testing ConvNeXtV2 MAE Model Creation:")
    x = torch.rand((2, 12, 256, 256))
    for size in ['small', 'base', 'large']:
        print(f"\nSize: {size}")
        model = create_convnextv2_mae(size=size, in_chans=12)
        loss, pred, mask = model(x)
        print(f"MAE Loss: {loss.item():.4f}, pred shape: {pred.shape}, mask shape: {mask.shape}")

    # Test segmentation model creation functions
    print("\nTesting ConvNeXtV2 Segmentation Model Creation:")
    x_seg = torch.rand((2, 3, 256, 256))
    for size in ['small', 'base', 'large']:
        print(f"\nSize: {size}")
        seg_model = create_convnextv2_segmentation(size=size, in_chans=3, num_classes=1)
        seg_output = seg_model(x_seg)
        print(f"Segmentation output shape: {seg_output.shape}")
