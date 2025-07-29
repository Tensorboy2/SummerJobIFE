from src.custom_dataset import get_dataloaders
from src.models.torch.convnextv2rms import create_convnextv2_mae, create_convnextv2_segmentation
from src.models.torch.vit import create_vit_mae, create_vit_segmentation
from src.train_mae import MAETrainer
from src.train_segmentation import SegmentationTrainer
import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==== Pretrain model ====

def run_mae_trainers():
    """
    Run MAE training for all model sizes and types (ViT, ConvNeXtV2).
    """
    mae_model_types = [
        #("vit", create_vit_mae, {"in_channels": 12, "patch_size": 16, "mask_ratio": 0.75}),
        ("convnextv2", create_convnextv2_mae, {"in_chans": 12, "mask_ratio": 0.75}),
    ]
    # sizes = ["atto","femto","pico","nano","small","base","large"]
    sizes = ["large","base","small"]
    for model_name, create_fn, extra_kwargs in mae_model_types:
        for size in sizes:
            print(f"\n=== Pretraining {model_name.upper()} MAE ({size}) ===")
            config = {
                'val_ratio': 0.2,
                'batch_size': 32,
                'data_type': 'mae',
                'lr': 0.0001,
                'weight_decay': 0.2,
                'num_epochs': 100,
                'decay': 'cosine',
                'warmup_steps': 2000,
                'use_amp': True,
                'compile': True,
                'max_grad_norm': 1.0,
                'specific_name': f'{model_name}_mae_{size}',
                'size': size,
            }
            # Merge size into kwargs
            kwargs = dict(extra_kwargs)
            kwargs["size"] = size
            model = create_fn(**kwargs).to(device=device)
            print(f"Model on device: {device}")
            train_loader, val_loader = get_dataloaders(config=config)
            trainer = MAETrainer(model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=device,
                                config=config)
            trainer.train()

def run_segmentation_trainers():
    """
    Run segmentation training for all model sizes and types (ViT, ConvNeXtV2).
    """
    seg_model_types = [
        ("convnextv2", create_convnextv2_segmentation, {"in_chans": 12, "num_classes": 1}),
        #("vit", create_vit_segmentation, {"in_channels": 12, "num_classes": 1, "patch_size": 16}),
    ]
    # sizes = ["atto","femto","pico","nano","small", "base"]
    sizes = ["atto"]

    for model_name, create_fn, extra_kwargs in seg_model_types:
        for size in sizes:
            print(f"\n=== Segmentation Training {model_name.upper()} ({size}) ===")
            config = {
                'val_ratio': 0.2,
                'batch_size': 16,  # Adjusted for segmentation
                'data_type': 'segmentation',
                'lr': 0.0001,
                'weight_decay': 0.2,
                'num_epochs': 100,
                'decay': 'cosine',
                'warmup_steps': 2000,
                'use_amp': True,
                'compile': True,
                'max_grad_norm': 1.0,
                'specific_name': f'{model_name}_seg_{size}',
                'size': size,
            }
            kwargs = dict(extra_kwargs)
            kwargs["size"] = size
            model = create_fn(**kwargs).to(device=device)
            # Optionally load encoder weights for segmentation (if available)
            import glob
            ckpt_pattern = os.path.join('checkpoints', 'mae', f'*{model_name}_mae_{size}*', f'encoder_*{model_name}_mae_{size}*_best.pt')
            ckpt_files = glob.glob(ckpt_pattern)
            
            print(f"Looking for pretrained encoder checkpoints: {ckpt_pattern}")
            if ckpt_files:
                encoder_ckpt_path = 'checkpoints/mae/ConvNeXtV2MAE_sizelarge_patch16_mask0.75_convnextv2_mae_large_20250729_055746/encoder_ConvNeXtV2MAE_sizelarge_patch16_mask0.75_convnextv2_mae_large_20250729_055746_best.pt'
                # encoder_ckpt_path = 'checkpoints/mae/ConvNeXtV2MAE_sizeatto_patch16_mask0.75_convnextv2_mae_atto_20250728_072844/encoder_ConvNeXtV2MAE_sizeatto_patch16_mask0.75_convnextv2_mae_atto_20250728_072844_best.pt'
                ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
                model.encoder.load_state_dict(ckpt['encoder'], strict=True)
                print(f"Loaded pretrained encoder from: {encoder_ckpt_path}")
            else:
                print(f"No pretrained encoder found for {model_name} {size}, training from scratch.")
            train_loader, val_loader = get_dataloaders(config=config)
            trainer = SegmentationTrainer(model=model,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         device=device,
                                         config=config)
            trainer.train()

# ==== Segmentation-train on  ====



# ==== Fine tune on FPV ====

def finetune():
    print(f'Started segmentation fine tuning...')
    # Implement as needed for your FPV fine-tuning scenario
    pass


import argparse

def main():
    parser = argparse.ArgumentParser(description="Model pipeline runner")
    parser.add_argument('--task', type=str, required=True, choices=['mae', 'segmentation', 'finetune'],
                        help='Which task to run: mae, segmentation, finetune')
    args = parser.parse_args()

    if args.task == 'mae':
        run_mae_trainers()
    elif args.task == 'segmentation':
        run_segmentation_trainers()
    elif args.task == 'finetune':
        finetune()
    else:
        print(f"Unknown task: {args.task}")

if __name__ == '__main__':
    main()
