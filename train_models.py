import torch
import numpy as np
import pandas as pd
import os

from models.unet import UNet
from models.convnextv2 import ConvNeXtV2Segmentation


from training_utils.metrics import *
from dataloader import get_dataloaders



def train_model():
    '''
    Function for training the U-Net or ConvNeXtV2 model on the dataset.
    It initializes the model, data loaders, loss function, optimizer, and training loop.
    Returns:
        - model: The trained model.
        - train_losses: List of training losses per epoch.
        - train_ious: List of training IoUs per epoch.
        - val_losses: List of validation losses per epoch.
        - val_ious: List of validation IoUs per epoch.
        - train_bces: List of training BCE losses per epoch.
        - val_bces: List of validation BCE losses per epoch.
        - train_dices: List of training Dice coefficients per epoch.
        - val_dices: List of validation Dice coefficients per epoch.
        - train_tps: List of training true positives per epoch.
        - val_tps: List of validation true positives per epoch.
        - train_fps: List of training false positives per epoch.
        - val_fps: List of validation false positives per epoch.
        - train_tns: List of training true negatives per epoch.
        - val_tns: List of validation true negatives per epoch.
        - train_fns: List of training false negatives per epoch.
        - val_fns: List of validation false negatives per epoch.
        - train_f1s: List of training F1 scores per epoch.
        - val_f1s: List of validation F1 scores per epoch.
        - train_recalls: List of training recalls per epoch.
        - val_recalls: List of validation recalls per epoch.
        - train_precisions: List of training precisions per epoch.
        - val_precisions: List of validation precisions per epoch.
    '''
    # Configuration
    config = {
        'model': 'unet',  # Choose between 'convnext' or 'unet'
        'batch_size': 8,
        'val_ratio': 0.2,
        'num_workers': 4,
        'learning_rate': 1e-4,  # Lower learning rate
        'num_epochs': 40,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Loss function weights - experiment with these!
        'loss_weights': {
            'bce': 1.0,     # Standard BCE
            'dice': 0.0,    # Dice loss for overlap
            'focal': 0.0,    # Focal loss for hard examples
        },
        'crop_size': (128, 128),  # Crop size for training augmentations
        'weight_decay': 0.1,  # Regularization
        'warmup_steps': 2000,  
        'learning_rate_decay': 'cosine',  # Use learning rate decay
        'plot_examples': False,  # Whether to plot examples during training
        'save_best_model': True  # Whether to save the best model based on validation Io

    }
    
    # Initialize model
    if config['model']=='convnext':
        model = ConvNeXtV2Segmentation(in_chans=12, 
                                       num_classes=1, 
                                       encoder_output_channels=320,
                                       open_model=False)
    elif config['model']=='unet':
        model = UNet(in_ch=12, out_ch=1)
    else:
        raise ValueError("Unsupported model type. Choose 'convnext' or 'unet'.")
    
    print(f"Using model: {model.name}")
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Loss and optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
    # Warmup scheduler: linearly increase LR for a few epochs, then use ReduceLROnPlateau

    def lr_lambda(step):
        if step < config.get('warmup_steps', 0):
            return float(step + 1) / float(config.get('warmup_steps', 1e-8)) # Linear warmup
        elif step > config.get('warmup_steps', 0) and config['learning_rate_decay'] == 'cosine':
            # Cosine decay after warmup:
            return 0.5 * (1 + np.cos(np.pi * (step - config.get('warmup_steps', 0)) / (config['num_epochs'] * len(train_loader) - config.get('warmup_steps', 0))))
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    train_losses = []
    train_ious = []
    train_bces = []
    train_dices = []
    train_tps = []
    train_fps = []
    train_tns = []
    train_fns = []
    train_f1s = []
    train_recalls = []
    train_precisions = []

    val_losses = []
    val_ious = []
    val_bces = []
    val_dices = []
    val_tps = []
    val_fps = []
    val_tns = []
    val_fns = []
    val_f1s = []
    val_recalls = []
    val_precisions = []

    best_val_iou = 0.0
    
    print(f"Training on device: {device}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    for epoch in range(config['num_epochs']):
        print('\nTraining...')
        model.train()
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        epoch_train_bce = 0.0
        epoch_train_dice = 0.0
        epoch_train_tp = 0
        epoch_train_fp = 0
        epoch_train_tn = 0
        epoch_train_fn = 0

        for train_batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, masks, config['loss_weights'])

            # Calculate metrics
            with torch.no_grad():
                iou = iou_score(outputs.float(), masks.float())
                bce = bce_loss(outputs, masks)
                dice = dice_coeff(outputs, masks)
                tp, fp, tn, fn = get_confusion_matrix(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            epoch_train_loss += loss.item()
            epoch_train_iou += iou.item()
            epoch_train_bce += bce.item()
            epoch_train_dice += dice.item()
            epoch_train_tp += tp
            epoch_train_fp += fp
            epoch_train_tn += tn
            epoch_train_fn += fn

            if train_batch_idx % 10 == 0:
                avg_loss = epoch_train_loss / (train_batch_idx + 1)
                avg_iou = epoch_train_iou / (train_batch_idx + 1)
                avg_bce = epoch_train_bce / (train_batch_idx + 1)
                avg_dice = epoch_train_dice / (train_batch_idx + 1)
                print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {train_batch_idx}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.8f}, BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f}")

        # Calculate average training metrics
        n_train_batches = train_batch_idx + 1
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_train_iou = epoch_train_iou / n_train_batches
        avg_train_bce = epoch_train_bce / n_train_batches
        avg_train_dice = epoch_train_dice / n_train_batches
        avg_train_tp = epoch_train_tp
        avg_train_fp = epoch_train_fp
        avg_train_tn = epoch_train_tn
        avg_train_fn = epoch_train_fn
        avg_train_precision = precision_score(avg_train_tp, avg_train_fp)
        avg_train_recall = recall_score(avg_train_tp, avg_train_fn)
        avg_train_f1 = f1_score(avg_train_tp, avg_train_fp, avg_train_fn)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_iou = 0.0
        epoch_val_bce = 0.0
        epoch_val_dice = 0.0
        epoch_val_tp = 0
        epoch_val_fp = 0
        epoch_val_tn = 0
        epoch_val_fn = 0
        print("\nValidating...")
        with torch.no_grad():
            for val_batch_idx , (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss = combined_loss(outputs, masks, config['loss_weights'])
                val_iou = iou_score(outputs, masks)
                bce = bce_loss(outputs, masks)
                dice = dice_coeff(outputs, masks)
                tp, fp, tn, fn = get_confusion_matrix(outputs, masks)

                epoch_val_loss += val_loss.item()
                epoch_val_iou += val_iou.item()
                epoch_val_bce += bce.item()
                epoch_val_dice += dice.item()
                epoch_val_tp += tp
                epoch_val_fp += fp
                epoch_val_tn += tn
                epoch_val_fn += fn

                if val_batch_idx % 10 == 0:
                    avg_loss = epoch_val_loss / (val_batch_idx + 1)
                    avg_iou = epoch_val_iou / (val_batch_idx + 1)
                    avg_bce = epoch_val_bce / (val_batch_idx + 1)
                    avg_dice = epoch_val_dice / (val_batch_idx + 1)
                    print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                        f"Batch {val_batch_idx}/{len(val_loader)}, "
                        f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.8f}, BCE: {avg_bce:.4f}, Dice: {avg_dice:.4f}")

        n_val_batches = val_batch_idx + 1
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_val_iou = epoch_val_iou / n_val_batches
        avg_val_bce = epoch_val_bce / n_val_batches
        avg_val_dice = epoch_val_dice / n_val_batches
        avg_val_tp = epoch_val_tp
        avg_val_fp = epoch_val_fp
        avg_val_tn = epoch_val_tn
        avg_val_fn = epoch_val_fn
        avg_val_precision = precision_score(avg_val_tp, avg_val_fp)
        avg_val_recall = recall_score(avg_val_tp, avg_val_fn)
        avg_val_f1 = f1_score(avg_val_tp, avg_val_fp, avg_val_fn)

        # Store history
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_bces.append(avg_train_bce)
        train_dices.append(avg_train_dice)
        train_tps.append(avg_train_tp)
        train_fps.append(avg_train_fp)
        train_tns.append(avg_train_tn)
        train_fns.append(avg_train_fn)
        train_f1s.append(avg_train_f1)
        train_recalls.append(avg_train_recall)
        train_precisions.append(avg_train_precision)

        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_bces.append(avg_val_bce)
        val_dices.append(avg_val_dice)
        val_tps.append(avg_val_tp)
        val_fps.append(avg_val_fp)
        val_tns.append(avg_val_tn)
        val_fns.append(avg_val_fn)
        val_f1s.append(avg_val_f1)
        val_recalls.append(avg_val_recall)
        val_precisions.append(avg_val_precision)

        # Save best model
        if avg_val_iou > best_val_iou and config.get('save_best_model', False):
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f'{model.name}_'+'best_unet_model.pth')
            print(f"New best model saved with validation IoU: {best_val_iou:.4f}")

        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train BCE: {avg_train_bce:.4f}, Train Dice: {avg_train_dice:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val BCE: {avg_val_bce:.4f}, Val Dice: {avg_val_dice:.4f}")
        print(f"Train TP: {avg_train_tp}, FP: {avg_train_fp}, TN: {avg_train_tn}, FN: {avg_train_fn}, F1: {avg_train_f1:.4f}, Recall: {avg_train_recall:.4f}, Precision: {avg_train_precision:.4f}")
        print(f"Val   TP: {avg_val_tp}, FP: {avg_val_fp}, TN: {avg_val_tn}, FN: {avg_val_fn}, F1: {avg_val_f1:.4f}, Recall: {avg_val_recall:.4f}, Precision: {avg_val_precision:.4f}")
        print("-" * 80)

    return model, train_losses, train_ious, val_losses, val_ious, train_bces, val_bces, train_dices, val_dices, train_tps, val_tps, train_fps, val_fps, train_tns, val_tns, train_fns, val_fns, train_f1s, val_f1s, train_recalls, val_recalls, train_precisions, val_precisions

if __name__ == "__main__":
    import pandas as pd
    (
        model, train_losses, train_ious, val_losses, val_ious,
        train_bces, val_bces, train_dices, val_dices,
        train_tps, val_tps, train_fps, val_fps, train_tns, val_tns, train_fns, val_fns,
        train_f1s, val_f1s, train_recalls, val_recalls, train_precisions, val_precisions
    ) = train_model()

    # Save metrics as a pandas DataFrame
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_ious,
        'val_iou': val_ious,
        'train_bce': train_bces,
        'val_bce': val_bces,
        'train_dice': train_dices,
        'val_dice': val_dices,
        'train_tp': train_tps,
        'val_tp': val_tps,
        'train_fp': train_fps,
        'val_fp': val_fps,
        'train_tn': train_tns,
        'val_tn': val_tns,
        'train_fn': train_fns,
        'val_fn': val_fns,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'train_recall': train_recalls,
        'val_recall': val_recalls,
        'train_precision': train_precisions,
        'val_precision': val_precisions
    })
    metrics_df.to_csv(os.path.join('results',f"{model.name}_metrics.csv"), index=False)
    print(f"Metrics saved to {model.name}_metrics.csv")