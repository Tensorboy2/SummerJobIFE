import torch.optim as op
from src.custom_dataset import get_dataloaders
# from src.models.torch.convnextv2 import ConvNeXtV2Segmentation, ConvNeXtV2MAE
# from src.models.torch.convnextv2rms import ConvNeXtV2Segmentation, ConvNeXtV2MAE
from src.models.torch.convnextv2rms import create_convnextv2_mae, create_convnextv2_segmentation
from src.train_mae import MAETrainer
from src.train_segmentation import SegmentationTrainer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
import tifffile

def load_s2_tiff_as_tensor(tiff_path):
    """
    Loads a Sentinel-2 TIFF, removes B10, returns torch tensor [C, H, W] float32.
    Assumes TIFF is [H, W, bands] and bands are in order B1-B12.
    """
    arr = tifffile.imread(tiff_path)  # [H, W, bands]
    # if arr.shape[-1] == 12:
    #     # Remove B10 (band index 9)
    #     arr = arr[..., [i for i in range(12) if i != 9]]  # [H, W, 11]
    if arr.shape[-1] == 13:
        # Some products have 13 bands (B1-B12 + extra)
        arr = arr[..., [i for i in range(13) if i != 9]]
    # Convert to torch tensor, channels first
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()*15
    return tensor

def main():
    import os
    image_dir = 'src/data/processed_unique/images'
    mask_dir = 'src/data/processed_unique/masks'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])

    # Load model
    model = create_convnextv2_segmentation(in_chans=12, num_classes=1, size='atto').to(device=device)
    # encoder_ckpt_path = "src/training_output/segmentation/pretrained_convnextv2rms.pt"
    encoder_ckpt_path = 'checkpoints/segmentation/convnextv2_seg_atto/best_model_convnextv2_seg_atto.pt'
    ckpt = torch.load(encoder_ckpt_path, map_location='cpu')['model_state_dict']
    # print(ckpt.keys())
    # model.encoder.load_state_dict(ckpt['encoder'], strict=False)
    # model.decoder.load_state_dict(ckpt['decoder'], strict=False)
    model.load_state_dict(ckpt, strict=False)
    print("Loaded pretrained encoder from:", encoder_ckpt_path)
    model = model.to(device=device)
    model.eval()

    # Example: load TIFFs and run inference (fixed path)
    tiff_root = 'src/google_earth_engine/downloaded_s2_annual_composites'
    if not os.path.isdir(tiff_root):
        print(f"TIFF directory not found: {tiff_root}")
    else:
        tiff_paths = [os.path.join(tiff_root, f) for f in os.listdir(tiff_root) if f.endswith('_multispectral.tif')]
        if not tiff_paths:
            print("No TIFF files found in the directory.")
        else:
            for i, tiff_path in enumerate(tiff_paths):
                img_tensor = load_s2_tiff_as_tensor(tiff_path)
                print(f"Loaded TIFF shape: {img_tensor.shape}")
                with torch.no_grad():
                    pred = model(img_tensor.unsqueeze(0).to(device=device))
                    pred = torch.sigmoid(pred).cpu()
                # Show RGB
                rgb = img_tensor[[3,2,1]].cpu().numpy().transpose(1,2,0)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                plt.imshow(rgb)
                plt.title('TIFF RGB')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(pred[0,0].numpy(), cmap='gray')
                plt.title('Pred')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'tiff_rgb_prediction_{os.path.basename(tiff_path)}.pdf', bbox_inches='tight')
                # plt.show()

    # Find examples with nonzero masks
    examples = []
    for img_file, mask_file in zip(image_files, mask_files):
        img = torch.load(os.path.join(image_dir, img_file))
        mask = torch.load(os.path.join(mask_dir, mask_file))
        if mask.sum() > 0:
            examples.append((img, mask, img_file))
        if len(examples) >= 2:
            break

    if not examples:
        print("No examples with nonzero masks found.")
        return

    n = len(examples)
    ncols = 3  # RGB, Prediction, Mask
    nrows = n
    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (img, mask, img_file) in enumerate(examples):
        with torch.no_grad():
            pred = model(img.unsqueeze(0).permute(0,3,1,2).to(device=device))
            pred = torch.sigmoid(pred).cpu()
        # RGB: Sentinel-2 B4, B3, B2 (channels 3,2,1)
        rgb = img[:,:,[3,2,1]].cpu().numpy()  # [3, H, W]
        # print(img.shape, rgb.shape, pred.shape, mask.shape)
        # rgb = rgb.transpose(1,2,0)  # [H, W, 3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)  # normalize for display
        plt.subplot(n, ncols, i*ncols+1)
        plt.imshow(rgb)
        plt.title(f'RGB: {img_file}')
        plt.axis('off')
        # Prediction
        plt.subplot(n, ncols, i*ncols+2)
        plt.imshow(pred[0,0].numpy(), cmap='gray')
        plt.title('Pred')
        plt.axis('off')
        # Mask
        plt.subplot(n, ncols, i*ncols+3)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('segmentation_examples.pdf', bbox_inches='tight')
    # plt.show()
if __name__ == '__main__':
    main()