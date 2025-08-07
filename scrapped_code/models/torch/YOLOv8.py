"""
YOLO Multi-Task Model for Object Detection and Segmentation

This module implements a multi-task YOLO model that performs both object detection
and semantic segmentation using a shared backbone and neck architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, activation: nn.Module = nn.SiLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class C2fBlock(nn.Module):
    """C2f block with multiple convolution paths and optional SE attention."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 n_blocks: int = 2, use_se: bool = True):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=1, 
                             stride=1, padding=0)
        self.blocks = nn.ModuleList([
            ConvBNAct(hidden_channels, hidden_channels) 
            for _ in range(n_blocks)
        ])
        self.fuse = ConvBNAct(hidden_channels * (n_blocks + 1), out_channels, 
                             kernel_size=1, padding=0)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        outputs = [x]
        
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
            
        x = self.fuse(torch.cat(outputs, dim=1))
        return self.se(x)


class SimpleYOLOBackbone(nn.Module):
    """
    Simple YOLO backbone network with three output scales.
    
    Args:
        in_channels: Number of input channels
        depth_mult: Depth multiplier for scaling number of blocks
        width_mult: Width multiplier for scaling channel dimensions
        use_coords: Whether to add coordinate channels to input
    """
    
    def __init__(self, in_channels: int = 3, depth_mult: float = 1.0, 
                 width_mult: float = 1.0, use_coords: bool = False):
        super().__init__()
        self.use_coords = use_coords

        def scale_channels(channels: int) -> int:
            return int(channels * width_mult)
        
        def scale_depth(depth: int) -> int:
            return max(1, int(depth * depth_mult))

        input_channels = in_channels + 2 if use_coords else in_channels

        # Network architecture
        self.stem = ConvBNAct(input_channels, scale_channels(16), 
                             kernel_size=3, stride=2)  # -> 64x64
        
        self.stage1 = C2fBlock(scale_channels(16), scale_channels(32), 
                              n_blocks=scale_depth(2))  # 64x64
        self.down1 = ConvBNAct(scale_channels(32), scale_channels(32), 
                              kernel_size=3, stride=2)  # -> 32x32
        
        self.stage2 = C2fBlock(scale_channels(32), scale_channels(64), 
                              n_blocks=scale_depth(2))  # 32x32
        self.down2 = ConvBNAct(scale_channels(64), scale_channels(64), 
                              kernel_size=3, stride=2)  # -> 16x16
        
        self.stage3 = C2fBlock(scale_channels(64), scale_channels(128), 
                              n_blocks=scale_depth(2))  # 16x16

    def add_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Add normalized coordinate channels to input tensor."""
        b, _, h, w = x.shape
        device = x.device
        
        y_coords = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Returns:
            List of feature maps: [P3, P4, P5] with shapes:
            - P3: [B, 32, 64, 64]
            - P4: [B, 64, 32, 32] 
            - P5: [B, 128, 16, 16]
        """
        if self.use_coords:
            x = self.add_coords(x)

        x = self.stem(x)
        x = self.stage1(x)
        p3 = x  # High resolution features
        
        x = self.down1(x)
        x = self.stage2(x)
        p4 = x  # Medium resolution features
        
        x = self.down2(x)
        x = self.stage3(x)
        p5 = x  # Low resolution features

        return [p3, p4, p5]


class AdvancedNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) style neck for multi-scale feature fusion.
    
    Args:
        channels: List of channel dimensions for [P3, P4, P5] features
    """
    
    def __init__(self, channels: List[int] = [32, 64, 128]):
        super().__init__()
        c3, c4, c5 = channels

        # Top-down pathway
        self.reduce_p5 = ConvBNAct(c5, c4, kernel_size=1, stride=1, padding=0)
        self.conv_p4 = ConvBNAct(c4 + c4, c4, kernel_size=3, stride=1)

        self.reduce_p4 = ConvBNAct(c4, c3, kernel_size=1, stride=1, padding=0)
        self.conv_p3 = ConvBNAct(c3 + c3, c3, kernel_size=3, stride=1)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass with top-down feature fusion.
        
        Args:
            features: List of [P3, P4, P5] feature maps
            
        Returns:
            List of fused multi-scale features
        """
        p3, p4, p5 = features

        # Top-down fusion: P5 -> P4 -> P3
        p5_upsampled = F.interpolate(self.reduce_p5(p5), scale_factor=2, mode='nearest')
        p4_fused = self.conv_p4(torch.cat([p4, p5_upsampled], dim=1))

        p4_upsampled = F.interpolate(self.reduce_p4(p4_fused), scale_factor=2, mode='nearest')
        p3_fused = self.conv_p3(torch.cat([p3, p4_upsampled], dim=1))

        return [p3_fused, p4_fused, p5]


class DetectionHead(nn.Module):
    """
    Multi-scale detection head for object detection.
    
    Args:
        channels: List of channel dimensions for different scales
        num_outputs: Number of output channels (bbox coords + objectness + classes)
    """
    
    def __init__(self, channels: List[int] = [32, 64, 128], num_outputs: int = 6):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c, kernel_size=3, stride=1),
                nn.Conv2d(c, num_outputs, kernel_size=1)
            ) for c in channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass returning detection outputs for each scale.
        
        Returns:
            List of detection tensors: [P3_det, P4_det, P5_det]
        """
        return [head(feat) for head, feat in zip(self.heads, features)]


class SegmentationHead(nn.Module):
    """
    U-Net style segmentation head with skip connections.
    
    Args:
        channels: List of channel dimensions for [P3, P4, P5] features
        output_size: Output segmentation map size
    """
    
    def __init__(self, channels: List[int] = [32, 64, 128], output_size: int = 128):
        super().__init__()
        c3, c4, c5 = channels

        # Decoder pathway
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
            nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass returning segmentation mask.
        
        Returns:
            Segmentation tensor: [B, 1, output_size, output_size]
        """
        p3, p4, p5 = features

        # Decoder with skip connections
        x = self.up1(p5)
        x = self.fuse1(torch.cat([x, p4], dim=1))  # 32x32

        x = self.up2(x)
        x = self.fuse2(torch.cat([x, p3], dim=1))  # 64x64

        return self.final_conv(x)


class YOLOMultiTask(nn.Module):
    """
    Multi-task YOLO model for simultaneous object detection and segmentation.
    
    Args:
        in_channels: Number of input channels
        input_size: Input image size
        backbone_config: Configuration for backbone scaling
    """
    
    def __init__(self, in_channels: int = 3, input_size: int = 256, 
                 backbone_config: Optional[Dict] = None):
        super().__init__()
        
        if backbone_config is None:
            backbone_config = {}
            
        self.backbone = SimpleYOLOBackbone(in_channels=in_channels, **backbone_config)
        self.neck = AdvancedNeck()
        self.detection_head = DetectionHead()
        self.segmentation_head = SegmentationHead(output_size=input_size)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass returning detection and segmentation outputs.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (detection_outputs, segmentation_output)
        """
        # Extract multi-scale features
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        
        # Generate outputs
        detection_outputs = self.detection_head(neck_features)
        segmentation_output = self.segmentation_head(neck_features)
        
        return detection_outputs, segmentation_output


# Utility Functions
def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert binary mask to bounding box coordinates.
    
    Args:
        mask: Binary mask tensor [H, W]
        
    Returns:
        Bounding box tensor [cx, cy, w, h] in normalized coordinates
    """
    y_indices, x_indices = torch.where(mask > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return torch.tensor([0.5, 0.5, 0.0, 0.0])  # No object found
    
    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()
    
    # Convert to normalized center coordinates and dimensions
    cx = (x1 + x2) / 2 / mask.shape[-1]
    cy = (y1 + y2) / 2 / mask.shape[-2]
    w = (x2 - x1) / mask.shape[-1]
    h = (y2 - y1) / mask.shape[-2]
    
    return torch.tensor([cx, cy, w, h])


def assign_targets(bbox: torch.Tensor, label: torch.Tensor, 
                  feature_size: int = 16, image_size: int = 256) -> torch.Tensor:
    """
    Assign ground truth targets to feature map grid cells.
    
    Args:
        bbox: Bounding boxes [B, 4] in normalized coordinates
        label: Class labels [B]
        feature_size: Size of feature map grid
        image_size: Size of input image
        
    Returns:
        Target tensor [B, 6, feature_size, feature_size]
    """
    batch_size = bbox.size(0)
    target_map = torch.zeros(batch_size, 6, feature_size, feature_size, device=bbox.device)

    for b in range(batch_size):
        x, y, w, h = bbox[b]
        
        # Convert to grid coordinates
        grid_x = int(x * feature_size)
        grid_y = int(y * feature_size)
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(grid_x, feature_size - 1))
        grid_y = max(0, min(grid_y, feature_size - 1))
        
        # Assign targets
        target_map[b, 0, grid_y, grid_x] = x  # Center x
        target_map[b, 1, grid_y, grid_x] = y  # Center y
        target_map[b, 2, grid_y, grid_x] = w  # Width
        target_map[b, 3, grid_y, grid_x] = h  # Height
        target_map[b, 4, grid_y, grid_x] = 1.0  # Objectness
        target_map[b, 5, grid_y, grid_x] = label[b]  # Class

    return target_map


def multitask_loss(detection_pred: List[torch.Tensor], segmentation_pred: torch.Tensor,
                  bbox: torch.Tensor, label: torch.Tensor, mask: torch.Tensor,
                  lambda_bbox: float = 5.0, lambda_seg: float = 1.0,
                  lambda_obj: float = 1.0, lambda_cls: float = 1.0,
                  scale_weights: List[float] = [1.0, 1.0, 1.0]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-task loss for detection and segmentation across multiple scales.
    
    Args:
        detection_pred: List of detection predictions from different scales
                       [[B, 6, H1, W1], [B, 6, H2, W2], [B, 6, H3, W3]]
        segmentation_pred: Segmentation predictions [B, 1, H, W]
        bbox: Ground truth bounding boxes [B, 4]
        label: Ground truth labels [B]
        mask: Ground truth segmentation masks [B, 1, H, W]
        lambda_*: Loss weighting factors
        scale_weights: Weights for different detection scales
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    total_cls_loss = 0.0
    
    # Compute detection losses for each scale
    for i, det_pred in enumerate(detection_pred):
        batch_size, _, height, width = det_pred.shape
        target_map = assign_targets(bbox, label, feature_size=height, image_size=256)

        assert target_map.shape == det_pred.shape, \
            f"Shape mismatch at scale {i}: pred {det_pred.shape}, target {target_map.shape}"

        # Object mask for focusing bbox loss on positive samples
        obj_mask = target_map[:, 4:5]

        # Compute individual losses for this scale
        bbox_loss = F.mse_loss(det_pred[:, 0:4] * obj_mask, target_map[:, 0:4] * obj_mask)
        obj_loss = F.binary_cross_entropy_with_logits(det_pred[:, 4:5], target_map[:, 4:5])
        cls_loss = F.binary_cross_entropy_with_logits(det_pred[:, 5:6], target_map[:, 5:6])
        
        # Weight by scale importance
        scale_weight = scale_weights[i] if i < len(scale_weights) else 1.0
        total_bbox_loss += scale_weight * bbox_loss
        total_obj_loss += scale_weight * obj_loss
        total_cls_loss += scale_weight * cls_loss
    
    # Average across scales
    num_scales = len(detection_pred)
    total_bbox_loss /= num_scales
    total_obj_loss /= num_scales
    total_cls_loss /= num_scales
    
    # Segmentation loss
    seg_loss = F.binary_cross_entropy_with_logits(segmentation_pred, mask)

    # Weighted total loss
    total_loss = (lambda_bbox * total_bbox_loss +
                  lambda_obj * total_obj_loss +
                  lambda_cls * total_cls_loss +
                  lambda_seg * seg_loss)

    loss_dict = {
        "total": total_loss.item(),
        "bbox": total_bbox_loss.item(),
        "objectness": total_obj_loss.item(),
        "classification": total_cls_loss.item(),
        "segmentation": seg_loss.item(),
    }

    return total_loss, loss_dict


# Example usage
if __name__ == '__main__':
    # Create model
    model = YOLOMultiTask(
        in_channels=12,
        input_size=256,
        backbone_config={
            'depth_mult': 1.0,
            'width_mult': 1.0,
            'use_coords': False
        }
    )
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 12, 256, 256)
    
    with torch.no_grad():
        detection_outputs, segmentation_output = model(input_tensor)
        
    print("Model created successfully!")
    print(f"Detection outputs: {len(detection_outputs)} scales")
    for i, det_out in enumerate(detection_outputs):
        print(f"  Scale {i}: {det_out.shape}")
    print(f"Segmentation output: {segmentation_output.shape}")