import torch
import matplotlib.pyplot as plt
import tifffile
import argparse
from u_net import UNet, ConvNeXtV2Segmentation

def load_s2_tiff_as_tensor(tiff_path):
    arr = tifffile.imread(tiff_path)
    if arr.shape[-1] == 13:
        arr = arr[..., [i for i in range(13) if i != 9]]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()*15
    return tensor

def main():
    import os
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description="Inference for UNet or ConvNeXtV2 models")
    parser.add_argument('--model', type=str, choices=['unet', 'convnextv2'], required=True, help='Model type to use')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to .pth state_dict file')
    parser.add_argument('--tiff_dir', type=str, default='src/google_earth_engine/downloaded_s2_annual_composites', help='Directory with TIFFs for inference')
    parser.add_argument('--n_examples', type=int, default=40, help='Number of dataset examples to visualize')
    args = parser.parse_args()

    image_dir = 'src/data/processed_unique/images'
    mask_dir = 'src/data/processed_unique/masks'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])

    # Load model
    if args.model == 'unet':
        model = UNet(in_ch=12, out_ch=1)
    elif args.model == 'convnextv2':
        model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1, encoder_output_channels=320)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load state_dict
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {args.model} weights from: {args.ckpt}")
    model = model.to(device)
    model.eval()

    # Inference on TIFFs
    tiff_root = args.tiff_dir
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
                    pred = model(img_tensor.unsqueeze(0).to(device))
                    pred = torch.sigmoid(pred).cpu() > 0.5
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
                plt.savefig(f'tiff_rgb_prediction_{os.path.basename(tiff_path)}_{args.model}.pdf', bbox_inches='tight')

    # Find examples with nonzero masks
    examples = []
    for img_file, mask_file in zip(image_files, mask_files):
        img = torch.load(os.path.join(image_dir, img_file))
        mask = torch.load(os.path.join(mask_dir, mask_file))
        if mask.sum() > 0:
            examples.append((img, mask, img_file))
        if len(examples) >= args.n_examples:
            break

    if not examples:
        print("No examples with nonzero masks found.")
        return

    n = len(examples)
    ncols = 3
    nrows = n
    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (img, mask, img_file) in enumerate(examples):
        with torch.no_grad():
            if img.shape[0] == 256 and img.shape[1] == 256 and img.shape[2] == 12:
                img_input = img.permute(2,0,1).unsqueeze(0)
            elif img.shape[0] == 12:
                img_input = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            pred = model(img_input.to(device))
            pred = torch.sigmoid(pred).cpu() >0.5
        if img.shape[0] == 12:
            rgb = img[[3,2,1]].cpu().numpy()
            rgb = rgb.transpose(1,2,0)
        else:
            rgb = img[:,:,[3,2,1]].cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        plt.subplot(n, ncols, i*ncols+1)
        plt.imshow(rgb)
        plt.title(f'RGB: {img_file}')
        plt.axis('off')
        plt.subplot(n, ncols, i*ncols+2)
        plt.imshow(pred[0,0].numpy(), cmap='gray')
        plt.title('Pred')
        plt.axis('off')
        plt.subplot(n, ncols, i*ncols+3)
        plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'segmentation_examples_{args.model}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()