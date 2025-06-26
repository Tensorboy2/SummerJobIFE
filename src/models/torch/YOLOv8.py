from models.torch.yolo_utils.backbone import SimpleYOLOBackbone
from models.torch.yolo_utils.neck import SimpleNeck
from models.torch.yolo_utils.heads import DetectionHead, SegmentationHead
import torch.nn as nn
class YOLOMultiTask(nn.Module):
    def __init__(self, in_ch=3, input_size=128):
        super().__init__()
        self.backbone = SimpleYOLOBackbone(in_ch=in_ch)
        self.neck = SimpleNeck(128, 128)

        self.det_head = DetectionHead(128, 6)    # bbox + object + class
        self.seg_head = SegmentationHead(128, input_size)

    def forward(self, x):
        features = self.neck(self.backbone(x))  # [B, 128, 16, 16]

        det_out = self.det_head(features)       # [B, 6, 16, 16]
        seg_out = self.seg_head(features)       # [B, 1, 128, 128]

        return det_out, seg_out
    


# if __name__ == '__main__':
    # model = YOLOMultiTask(in_ch=12,input_size=256)
    # data = 