import torch.nn as nn
import torch

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

class C2fBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2):
        super().__init__()
        hidden_ch = out_ch // 2
        self.stem = ConvBNAct(in_ch, hidden_ch, k=1, s=1, p=0)

        self.blocks = nn.Sequential(*[
            ConvBNAct(hidden_ch, hidden_ch) for _ in range(n_blocks)
        ])

        self.fuse = ConvBNAct(hidden_ch * (n_blocks + 1), out_ch, k=1, p=0)

    def forward(self, x):
        x = self.stem(x)
        outputs = [x]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return self.fuse(torch.cat(outputs, dim=1))

class SimpleYOLOBackbone(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 16, k=3, s=2)       # 64x64
        self.stage1 = C2fBlock(16, 32)                   # 64x64
        self.down1 = ConvBNAct(32, 32, k=3, s=2)         # 32x32

        self.stage2 = C2fBlock(32, 64)                   # 32x32
        self.down2 = ConvBNAct(64, 64, k=3, s=2)         # 16x16

        self.stage3 = C2fBlock(64, 128)                  # 16x16

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return x  # [B, 128, 16, 16]

if __name__ == '__main__':
    model = SimpleYOLOBackbone()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print("Output shape:", out.shape)  # should be [1, 128, 16, 16]
