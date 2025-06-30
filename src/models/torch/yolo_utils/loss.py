import torch.nn.functional as F
# from models.torch.yolo_utils.heads import assign_targets
from models.torch.yolo_utils.data import PTMultiTaskDataset
import torch
def mask_to_bbox(mask):
    y_indices, x_indices = torch.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return torch.tensor([0.5, 0.5, 0.0, 0.0])  # no object
    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()
    cx = (x1 + x2) / 2 / mask.shape[-1]
    cy = (y1 + y2) / 2 / mask.shape[-2]
    w = (x2 - x1) / mask.shape[-1]
    h = (y2 - y1) / mask.shape[-2]
    return torch.tensor([cx, cy, w, h])


def assign_targets(bbox, label, feat_size=16, img_size=256):
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

def multitask_loss(det_pred, seg_pred, bbox, label, mask, 
                   lambda_bbox=1, lambda_seg=5, lambda_obj=1, lambda_cls=1):

    B, _, Hf, Wf = det_pred.shape
    target_map = assign_targets(bbox, label, feat_size=Hf, img_size=256)

    assert target_map.shape == det_pred.shape, \
        f"Mismatch: pred {det_pred.shape}, target {target_map.shape}"

    obj_mask = target_map[:, 4:5]

    bbox_loss = F.mse_loss(det_pred[:, 0:4] * obj_mask, target_map[:, 0:4] * obj_mask)
    obj_loss  = F.binary_cross_entropy_with_logits(det_pred[:, 4:5], target_map[:, 4:5])
    cls_loss  = F.binary_cross_entropy_with_logits(det_pred[:, 5:6], target_map[:, 5:6])
    seg_loss  = F.binary_cross_entropy_with_logits(seg_pred, mask)

    total = (lambda_bbox * bbox_loss +
             lambda_obj  * obj_loss +
             lambda_cls  * cls_loss +
             lambda_seg  * seg_loss)

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
