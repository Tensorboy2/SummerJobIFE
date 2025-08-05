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

class ConvNeXtV3Stem(nn.Module):
    '''
    Stem layer for ConvNeXt v2.
    '''
    def __init__(self, in_chans=3, out_chans=96):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_chans, 
                                   in_chans, 
                                   kernel_size=4, 
                                   stride=4, 
                                   padding=0, 
                                   groups=in_chans, 
                                   bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.norm = RMSNorm2d(out_chans, eps=1e-6)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x

class ConvNeXtV3Encoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3,3,9,3], dims=[96,192,384,768]):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        stem = ConvNeXtV3Stem(in_chans, dims[0])
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

class ConvNeXtV3Decoder(nn.Module):
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

class ConvNeXtV3SegmentationDecoder(nn.Module):
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


class ConvNeXtV3Segmentation(nn.Module):
    def __init__(self, in_chans=3, num_classes=21, depths=[3,3,9,3], dims=[96,192,384,768],
                 decoder_dims=[384,192,96,48]):
        super().__init__()
        self.name = 'ConvNeXtV3'
        self.encoder = ConvNeXtV3Encoder(in_chans, depths, dims)
        self.decoder = ConvNeXtV3SegmentationDecoder(dims, decoder_dims, num_classes)

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
        _, encoder_features = self.encoder(x)
        return self.decoder(encoder_features)

class ConvNeXtV3(ConvNeXtV3Encoder):
    def forward(self, x):
        latent, features = super().forward(x)
        return latent

# --- Model creation functions for different sizes (like ViT) ---
def create_convnextv3_segmentation(size='base', in_chans=3, num_classes=1):
    """Create a ConvNeXtV3 model for segmentation tasks with different sizes."""
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
            'dims': [16, 32, 64, 128],  # Stem will output 16 channels for atto
            'decoder_dims': [64, 32, 16, 8]
        }
    }
    if size not in configs:
        raise ValueError(f"Unsupported ConvNeXtV3 size: {size}. Choose from {list(configs.keys())}")
    cfg = configs[size]
    return ConvNeXtV3Segmentation(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=cfg['depths'],
        dims=cfg['dims'],
        decoder_dims=cfg['decoder_dims']
    )

if __name__ == '__main__':
    # Test segmentation model creation functions
    print("\nTesting ConvNeXtV3 Segmentation Model Creation:")
    x_seg = torch.rand((2, 12, 256, 256))
    for size in ['atto']:
        print(f"\nSize: {size}")
        seg_model = create_convnextv3_segmentation(size=size, in_chans=12, num_classes=1)
        seg_output = seg_model(x_seg)
        print(f"Segmentation output shape: {seg_output.shape}")
