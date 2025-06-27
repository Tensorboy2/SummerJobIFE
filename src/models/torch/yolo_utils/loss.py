import torch.nn.functional as F
from models.torch.yolo_utils.heads import assign_targets
from models.torch.yolo_utils.data import PTMultiTaskDataset

def multitask_loss(det_pred, seg_pred, bbox, label, mask, 
                   lambda_bbox=5, lambda_seg=1, lambda_obj=1, lambda_cls=1):

    B, _, Hf, Wf = det_pred.shape
    target_map = assign_targets(bbox, label, feat_size=Hf, img_size=128)

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
