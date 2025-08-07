import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torch.yolo_utils.backbone import ConvBNAct
# from backbone import ConvBNAct

# 🔍 Multi-scale Detection Head
class DetectionHead(nn.Module):
    def __init__(self, channels=[32, 64, 128], num_outputs=6):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c, k=3, s=1),
                nn.Conv2d(c, num_outputs, kernel_size=1)
            ) for c in channels
        ])

    def forward(self, feats):
        # feats: list of [P3, P4, P5]
        return [head(f) for head, f in zip(self.heads, feats)]
        # Returns list: [[B, 6, 64, 64], [B, 6, 32, 32], [B, 6, 16, 16]]



# 🧼 Advanced Segmentation Decoder Head (U-Net-style with skip fusion)
class SegmentationHead(nn.Module):
    def __init__(self, channels=[32, 64, 128], out_size=128):
        super().__init__()
        c3, c4, c5 = channels

        self.up1 = nn.Sequential(
            nn.Conv2d(c5, c4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c4 + c4, c4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(c4, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(c3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False)
        )

    def forward(self, feats):
        # feats: [P3, P4, P5]
        p3, p4, p5 = feats

        x = self.up1(p5)
        x = self.fuse1(torch.cat([x, p4], dim=1))  # 32x32

        x = self.up2(x)
        x = self.fuse2(torch.cat([x, p3], dim=1))  # 64x64

        return self.final_conv(x)  # [B, 1, 128, 128]

if __name__ == '__main__':
    B = 2
    feats = [torch.randn(B, 32, 64, 64), torch.randn(B, 64, 32, 32), torch.randn(B, 128, 16, 16)]

    det_head = DetectionHead()
    det_outs = det_head(feats)
    for i, o in enumerate(det_outs):
        print(f"Det scale {i}: {o.shape}")

    seg_head = SegmentationHead()
    seg_out = seg_head(feats)
    print("Segmentation output:", seg_out.shape)
