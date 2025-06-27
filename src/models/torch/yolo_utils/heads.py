from models.torch.yolo_utils.backbone import ConvBNAct
import torch.nn as nn
import torch

class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_outputs=6):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3, s=1),
            nn.Conv2d(in_ch, num_outputs, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)  # [B, 6, 16, 16]

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, out_size=256):
        super().__init__()
        self.seg = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output mask in [0, 1]
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False)  # final resize
        )

    def forward(self, x):
        return self.seg(x)  # [B, 1, 128, 128]





def assign_targets(bbox, label, feat_size=16, img_size=128):
    # bbox: [B, 4] in normalized coords
    B = bbox.size(0)
    target_map = torch.zeros(B, 6, feat_size, feat_size, device=bbox.device)

    for b in range(B):
        x, y, w, h = bbox[b]
        cx = int(x * feat_size)
        cy = int(y * feat_size)

        target_map[b, 0, cy, cx] = x
        target_map[b, 1, cy, cx] = y
        target_map[b, 2, cy, cx] = w
        target_map[b, 3, cy, cx] = h
        target_map[b, 4, cy, cx] = 1.0  # objectness
        target_map[b, 5, cy, cx] = label[b]  # class (0 or 1)

    return target_map  # [B, 6, 16, 16]



# if __name__ == '__main__':
#     # model = YOLOMultiTask(in_ch=12,input_size=256)
#     x = torch.randn(2, 12, 256, 256)
#     # det_out, seg_out = model(x)

#     print("Detection output:", det_out.shape)  # [2, 6, 16, 16]
#     print("Segmentation output:", seg_out.shape)  # [2, 1, 128, 128]
