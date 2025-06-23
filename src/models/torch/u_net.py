import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''
    Bottleneck block for U-net
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class UNetWithSkips(nn.Module):
    '''
    U-net with skip connections.
    '''
    def __init__(self, in_ch=12, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))

        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.final(x)

class MiniUNet(nn.Module):
    def __init__(self, in_ch=12, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 16)
        self.enc2 = DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = DoubleConv(32, 16)

        self.final = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)          # [B, 16, H, W]
        x2 = self.enc2(self.pool(x1))  # [B, 32, H/2, W/2]
        x3 = self.bottleneck(self.pool(x2))  # [B, 64, H/4, W/4]

        x = self.up1(x3)           # [B, 32, H/2, W/2]
        x = self.dec1(torch.cat([x, x2], dim=1))

        x = self.up2(x)            # [B, 16, H, W]
        x = self.dec2(torch.cat([x, x1], dim=1))

        return self.final(x)