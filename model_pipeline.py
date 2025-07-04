import torch.optim as op
from src.custom_dataset import get_dataloaders
from src.models.torch.convnextv2 import ConvNeXtV2Segmentation, ConvNeXtV2MAE
# from src.models.torch.convnextv2rms import ConvNeXtV2Segmentation, ConvNeXtV2MAE
from src.train_mae import MAETrainer
from src.train_segmentation import SegmentationTrainer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==== Pretrain model ====
def pretrain():
    '''
    Pretrain ConvNeXt v2 using fully convolutional masked auto encoder.
    On full images from Yang dataset. (all years)
    '''
    print(f'Started pre-training...')
    print(f'Using {device}...')

    config = {
        'val_ratio':0.2,
        'batch_size':128,
        'data_type':'mae',
        'lr':0.0001,
        'weight_decay':0.1,
        'num_epochs':100,
        'decay': 'cosine',
        'warmup_steps': 500,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
        'specific_name':'convnextv2ln',
    }
    model = ConvNeXtV2MAE(mask_ratio=0.6).to(device=device)
    train_loader, val_loader = get_dataloaders(config=config)
    trainer = MAETrainer(model=model,
               train_loader=train_loader,
               val_loader=val_loader,
               device=device,
               config=config)
    trainer.train()
    return 0

# ==== Segmentation-train on  ====
def segmentation_train():
    '''
    Train ConvNeXt v2 on segmentation data using metrics such as:
    - dice
    - iou

    Masks are part of the Yang dataset.
    '''
    print(f'Started segmentation-training...')
    config = {
        'val_ratio':0.2,
        'batch_size':32,
        'data_type':'segmentation',
        'lr':0.0001,
        'weight_decay':0.1,
        'num_epochs':100,
        'decay': 'cosine',
        'warmup_steps': 500,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
        'specific_name':'convnextv2ln',
    }
    model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1)
    encoder_ckpt_path = "pretrained_encoder_convnextv2ln.pt"
    ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
    model.encoder.load_state_dict(ckpt['encoder'], strict=True)
    print("Loaded pretrained encoder from:", encoder_ckpt_path)

    train_loader, val_loader = get_dataloaders(config=config)

    trainer = SegmentationTrainer(model=model,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 device=device,
                                 config=config)
    trainer.train()

    return 0


# ==== Fine tune on FPV ====
def finetune():
    '''
    Must create samples of FPV systems based on mask from Xia or dataset from Xia.
    These samples are used to fine tuned the model for FPVs.
    '''
    print(f'Started segmentation fine tuning...')

    config = {
        'val_ratio':0.2,
        'batch_size':32,
        'data_type':'segmentation',
        'lr':1e-4,
        'weight_decay':0.5,
        'num_epochs':100,
        'decay': 'cosine',
        'warmup_steps': 1000,
        'use_amp': True,
        'compile': True,
        'max_grad_norm': 1.0,
    }
    model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1)
    train_loader, val_loader = get_dataloaders(config=config)
    return 0


if __name__ == '__main__':
    pretrain()
    segmentation_train()
    finetune()
