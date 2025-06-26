from torch.utils.data import DataLoader
import torch.optim as optim
from models.torch.yolo_utils.data import RandomMultiTaskDataset
from heads import YOLOMultiTask
from loss import multitask_loss
from collections import defaultdict
import torch

import matplotlib.pyplot as plt

def visualize_prediction(model, dataset, device="cpu"):
    model.eval()
    img, mask, bbox, label = dataset[0]
    
    with torch.no_grad():
        img_in = img.unsqueeze(0).to(device)
        det_out, seg_out = model(img_in)
    
    seg_pred = seg_out.squeeze().cpu().numpy()
    det = det_out.squeeze().cpu()

    # Find the highest objectness score
    obj = torch.sigmoid(det[4])  # objectness map
    max_idx = torch.argmax(obj)
    y, x = divmod(max_idx.item(), obj.size(1))  # (row, col)

    pred_box = det[:4, y, x]  # (x, y, w, h) in normalized
    pred_class = torch.sigmoid(det[5, y, x]).item()

    # Denormalize
    H, W = 128, 128
    px = pred_box[0].item() * W
    py = pred_box[1].item() * H
    pw = pred_box[2].item() * W
    ph = pred_box[3].item() * H

    # Plot
    img_np = img.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.gca().add_patch(plt.Rectangle((px - pw/2, py - ph/2), pw, ph, edgecolor='lime', facecolor='none'))
    plt.text(px, py, f"{pred_class:.2f}", color='white', bbox=dict(facecolor='black'))

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(0), cmap='gray')
    plt.title("True Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(seg_pred, cmap='gray')
    plt.title("Predicted Mask")

    plt.show()

def train_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    running_loss = defaultdict(float)
    
    for img, mask, bbox, label in dataloader:
        img = img.to(device)
        mask = mask.to(device)
        bbox = bbox.to(device)
        label = label.to(device)

        det_out, seg_out = model(img)
        loss, logs = multitask_loss(det_out, seg_out, bbox, label, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in logs.items():
            running_loss[k] += v

    avg_loss = {k: v / len(dataloader) for k, v in running_loss.items()}
    return avg_loss

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


if __name__ == '__main__':
    # Dataset
    train_ds = RandomMultiTaskDataset(length=500)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

    # Model and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOMultiTask().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        logs = train_epoch(model, train_dl, optimizer, device)
        print(f"Epoch {epoch}: {logs}")
        if epoch % 2 == 0:
            visualize_prediction(model, train_ds, device)