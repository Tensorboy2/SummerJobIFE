import torch
import torch.nn as nn
import torch.nn.functional as F

def iou_score(preds, targets, threshold=0.5):
    """Calculate IoU score for batch of predictions and targets"""
    # Apply sigmoid to predictions if they're logits
    if preds.max() > 1 or preds.min() < 0:
        preds = torch.sigmoid(preds)
    
    preds = (preds > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    
    # Avoid division by zero
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean()

def dice_loss(preds, targets, smooth=1e-8):
    """Dice loss for better segmentation training"""
    preds = torch.sigmoid(preds)
    
    intersection = (preds * targets).sum(dim=(2, 3))
    dice_coeff = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return 1 - dice_coeff.mean()

def focal_loss(preds, targets, alpha=1, gamma=2, smooth=1e-8):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Weighting factor for rare class (default: 1)
        gamma: Focusing parameter (default: 2)
    """
    # Apply sigmoid to get probabilities
    prob = torch.sigmoid(preds)
    
    # Calculate BCE loss
    bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
    
    # Calculate p_t
    p_t = prob * targets + (1 - prob) * (1 - targets)
    
    # Calculate alpha_t
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Calculate focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss = focal_weight * bce_loss
    
    return focal_loss.mean()

def combined_loss(preds, targets, loss_weights=None):
    """
    Combined loss function with multiple options
    
    Args:
        loss_weights: Dict with keys 'bce', 'dice', 'focal' and their respective weights
    """
    if loss_weights is None:
        loss_weights = {'bce': 0.5, 'dice': 0.5, 'focal': 0.0}
    
    total_loss = 0.0
    
    if loss_weights.get('bce', 0) > 0:
        bce = nn.BCEWithLogitsLoss()(preds, targets)
        total_loss += loss_weights['bce'] * bce
    
    if loss_weights.get('dice', 0) > 0:
        dice = dice_loss(preds, targets)
        total_loss += loss_weights['dice'] * dice
    
    if loss_weights.get('focal', 0) > 0:
        focal = focal_loss(preds, targets, alpha=0.75, gamma=2)
        total_loss += loss_weights['focal'] * focal
    
    return total_loss

# --- Metric helpers ---
def bce_loss(preds, targets):
    return nn.BCEWithLogitsLoss()(preds, targets)

def dice_coeff(preds, targets, smooth=1e-8):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return dice.mean()

def get_confusion_matrix(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    return tp, fp, tn, fn

def precision_score(tp, fp):
    return tp / (tp + fp + 1e-8)

def recall_score(tp, fn):
    return tp / (tp + fn + 1e-8)

def f1_score(tp, fp, fn):
    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)
    return 2 * (prec * rec) / (prec + rec + 1e-8)