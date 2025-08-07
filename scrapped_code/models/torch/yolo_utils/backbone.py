import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class C2fBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2, use_se=True):
        super().__init__()
        hidden_ch = out_ch // 2
        self.stem = ConvBNAct(in_ch, hidden_ch, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*[
            ConvBNAct(hidden_ch, hidden_ch) for _ in range(n_blocks)
        ])
        self.fuse = ConvBNAct(hidden_ch * (n_blocks + 1), out_ch, k=1, p=0)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        outputs = [x]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        x = self.fuse(torch.cat(outputs, dim=1))
        return self.se(x)

class SimpleYOLOBackbone(nn.Module):
    def __init__(self, in_ch=3, depth_mult=1.0, width_mult=1.0, use_coords=False):
        super().__init__()
        self.use_coords = use_coords

        def ch(val): return int(val * width_mult)
        def depth(val): return max(1, int(val * depth_mult))

        input_ch = in_ch + 2 if use_coords else in_ch

        self.stem = ConvBNAct(input_ch, ch(16), k=3, s=2)              # 64x64
        self.stage1 = C2fBlock(ch(16), ch(32), n_blocks=depth(2))     # 64x64
        self.down1 = ConvBNAct(ch(32), ch(32), k=3, s=2)              # 32x32

        self.stage2 = C2fBlock(ch(32), ch(64), n_blocks=depth(2))     # 32x32
        self.down2 = ConvBNAct(ch(64), ch(64), k=3, s=2)              # 16x16

        self.stage3 = C2fBlock(ch(64), ch(128), n_blocks=depth(2))    # 16x16

    def add_coords(self, x):
        b, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x):
        if self.use_coords:
            x = self.add_coords(x)

        x = self.stem(x)
        x = self.stage1(x)
        out1 = x                      # P3: [B, 32, 64, 64]
        x = self.down1(x)

        x = self.stage2(x)
        out2 = x                      # P4: [B, 64, 32, 32]
        x = self.down2(x)

        x = self.stage3(x)
        out3 = x                      # P5: [B, 128, 16, 16]

        return [out1, out2, out3]

# Test
if __name__ == '__main__':
    model = SimpleYOLOBackbone(in_ch=3, depth_mult=1.0, width_mult=1.0, use_coords=False)
    x = torch.randn(1, 3, 128, 128)
    feats = model(x)
    for i, f in enumerate(feats):
        print(f"P{i+3} shape: {f.shape}")
