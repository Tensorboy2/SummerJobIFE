import torch.nn.functional as F
from heads import assign_targets,YOLOMultiTask
from models.torch.yolo_utils.data import RandomMultiTaskDataset

def multitask_loss(det_pred, seg_pred, bbox, label, mask, lambda_bbox=5, lambda_seg=1, lambda_obj=1, lambda_cls=1):
    """
    det_pred: [B, 6, 16, 16]
    seg_pred: [B, 1, 128, 128]
    bbox:     [B, 4]
    label:    [B]
    mask:     [B, 1, 128, 128]
    """

    # === Detection loss ===
    B = det_pred.size(0)
    target_map = assign_targets(bbox, label, feat_size=16, img_size=128)
    obj_mask = target_map[:, 4:5]  # where objectness = 1

    # Bounding box (only where obj_mask == 1)
    bbox_loss = F.mse_loss(det_pred[:, 0:4] * obj_mask, target_map[:, 0:4] * obj_mask)

    # Objectness (BCE)
    obj_loss = F.binary_cross_entropy_with_logits(det_pred[:, 4:5], target_map[:, 4:5])

    # Class loss (BCE, binary classification)
    cls_loss = F.binary_cross_entropy_with_logits(det_pred[:, 5:6], target_map[:, 5:6])

    # === Segmentation loss ===
    seg_loss = F.binary_cross_entropy(seg_pred, mask)

    # === Total loss ===
    total = (
        lambda_bbox * bbox_loss +
        lambda_obj * obj_loss +
        lambda_cls * cls_loss +
        lambda_seg * seg_loss
    )

    return total, {
        "total": total.item(),
        "bbox": bbox_loss.item(),
        "obj": obj_loss.item(),
        "cls": cls_loss.item(),
        "seg": seg_loss.item(),
    }


if __name__ == '__main__':
    # Dummy inputs
    model = YOLOMultiTask()
    data = RandomMultiTaskDataset()
    img, mask, bbox, label = data[0]

    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    bbox = bbox.unsqueeze(0)
    label = label.unsqueeze(0)

    det_out, seg_out = model(img)
    loss, logs = multitask_loss(det_out, seg_out, bbox, label, mask)

    print("Loss:", logs)
