import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torch.yolo_utils.backbone import ConvBNAct, SimpleYOLOBackbone
# from backbone import ConvBNAct, SimpleYOLOBackbone

class AdvancedNeck(nn.Module):
    def __init__(self, channels=[32, 64, 128]):  # P3, P4, P5 channels
        super().__init__()
        c3, c4, c5 = channels

        # Reduce C5 to match C4
        self.reduce_p5 = ConvBNAct(c5, c4, k=1, s=1, p=0)
        self.conv_p4 = ConvBNAct(c4 + c4, c4, k=3, s=1)

        # Reduce merged P4 to match C3
        self.reduce_p4 = ConvBNAct(c4, c3, k=1, s=1, p=0)
        self.conv_p3 = ConvBNAct(c3 + c3, c3, k=3, s=1)

        # Optional extra processing on P5
        self.down_p5 = ConvBNAct(c5, c5, k=3, s=2)

    def forward(self, features):
        # Unpack: P3 = high res, P5 = low res
        p3, p4, p5 = features  # [B, c3, 64, 64], [B, c4, 32, 32], [B, c5, 16, 16]

        # Top-down fusion
        p5_up = F.interpolate(self.reduce_p5(p5), scale_factor=2, mode='nearest')
        p4 = self.conv_p4(torch.cat([p4, p5_up], dim=1))  # [B, c4, 32, 32]

        p4_up = F.interpolate(self.reduce_p4(p4), scale_factor=2, mode='nearest')
        p3 = self.conv_p3(torch.cat([p3, p4_up], dim=1))  # [B, c3, 64, 64]

        return [p3, p4, p5]  # multi-scale outputs for detection/segmentation


class BackboneWithNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleYOLOBackbone()
        self.neck = AdvancedNeck(channels=[32, 64, 128])

    def forward(self, x):
        feats = self.backbone(x)         # [P3, P4, P5]
        fused_feats = self.neck(feats)   # [P3', P4', P5']
        return fused_feats


# Test
if __name__ == '__main__':
    model = BackboneWithNeck()
    x = torch.randn(1, 3, 128, 128)
    feats = model(x)
    for i, f in enumerate(feats):
        print(f"Fused P{i+3} shape:", f.shape)
