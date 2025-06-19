import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- Building Blocks ----

class DoubleConv(nn.Module):
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

class UNetWithoutSkips(nn.Module):
    def __init__(self, in_ch=12, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))

        x = self.up3(x4)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.final(x)

# ---- Fake Training Setup ----

def make_fake_batch(batch_size=2, H=256, W=256, C=12):
    """Generates random input and random circular blob masks"""
    x = torch.randn(batch_size, C, H, W)
    y = torch.zeros(batch_size, 1, H, W)
    for i in range(batch_size):
        center = torch.randint(64, H - 64, (2,))
        radius = torch.randint(20, 60, (1,))
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((yy - center[0])**2 + (xx - center[1])**2) < radius**2
        y[i, 0] = mask.float()
    return x, y

def train_one_step(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    optimizer.step()
    return loss.item(), out

# ---- Comparison ----

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# models = {
#     "WithSkips": UNetWithSkips().to(device),
#     "NoSkips": UNetWithoutSkips().to(device),
# }

# losses = {}

# for name, model in models.items():
#     optim = torch.optim.Adam(model.parameters(), lr=1e-3)
#     print(f"\nTraining model: {name}")
#     model_losses = []
#     for step in range(10):  # just 10 steps for quick test
#         x, y = make_fake_batch()
#         x, y = x.to(device), y.to(device)
#         loss, _ = train_one_step(model, x, y, optim)
#         model_losses.append(loss)
#         print(f"Step {step+1:02d}: Loss = {loss:.4f}")
#     losses[name] = model_losses

# # ---- Plot loss curves ----
# plt.plot(losses["WithSkips"], label="With Skips")
# plt.plot(losses["NoSkips"], label="No Skips")
# plt.xlabel("Training Step")
# plt.ylabel("Loss")
# plt.title("Loss Comparison (Skip vs No Skip U-Net)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
