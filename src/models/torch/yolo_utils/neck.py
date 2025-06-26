from yolo_utils.backbone import ConvBNAct, SimpleYOLOBackbone
import torch.nn as nn
import torch

class SimpleNeck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=3, s=1),
            ConvBNAct(out_ch, out_ch, k=3, s=1)
        )

    def forward(self, x):
        return self.fuse(x)  # keeps same shape

class BackboneWithNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleYOLOBackbone()
        self.neck = SimpleNeck(in_ch=128, out_ch=128)

    def forward(self, x):
        features = self.backbone(x)        # [B, 128, 16, 16]
        fused = self.neck(features)        # [B, 128, 16, 16]
        return fused


# if __name__ == '__main__':
#     model = BackboneWithNeck()
#     x = torch.randn(1, 3, 128, 128)
#     out = model(x)
#     print("Neck Output shape:", out.shape)  # should be [1, 128, 16, 16]
