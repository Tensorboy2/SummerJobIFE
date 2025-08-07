import torch
import torch.nn as nn


import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, encoder_output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(encoder_output_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv(x)  # [B, 1, 8, 8]
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x
    

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        # x is [B, C, H, W]
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
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
        mmearth_model = torch.hub.load('vishalned/mmearth-train', 'MPMAE', model_name='convnextv2_atto', pretrained=True, linear_probe=True)
        encoder = ConvNeXtV2Encoder(mmearth_model)
        self.encoder = encoder
        self.decoder = Decoder(encoder_output_channels=encoder_output_channels)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = ConvNeXtV2Segmentation()

x = torch.randn(1, 12, 256, 256)  # Example input tensor
output = model(x)
print("Output shape:", output.shape)  # Should be [1, 1,