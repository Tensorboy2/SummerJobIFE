import torch
import matplotlib.pyplot as plt
import tifffile
import argparse
import numpy as np
from u_net import UNet, ConvNeXtV2Segmentation

def get_model(model_name, ckpt_path, device):
    if model_name == 'unet':
        model = UNet(in_ch=12, out_ch=1)
    elif model_name.startswith('convnextv2'):
        model = ConvNeXtV2Segmentation(in_chans=12, num_classes=1, encoder_output_channels=320)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def run_inference(model, img, device):
    with torch.no_grad():
        if img.shape[0] == 256 and img.shape[1] == 256 and img.shape[2] == 12:
            img_input = img.permute(2,0,1).unsqueeze(0)
        elif img.shape[0] == 12:
            img_input = img.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        pred_logits = model(img_input.to(device))
        pred_probs = torch.sigmoid(pred_logits).cpu()
        pred_binary = (pred_probs > 0.5).float()
    return pred_binary[0,0].numpy(), pred_probs[0,0].numpy()

def get_confusion_map(mask_np, pred_binary):
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    pred_bin = (pred_binary > 0.5).astype(np.uint8)
    tp = np.sum((mask_bin == 1) & (pred_bin == 1))
    tn = np.sum((mask_bin == 0) & (pred_bin == 0))
    fp = np.sum((mask_bin == 0) & (pred_bin == 1))
    fn = np.sum((mask_bin == 1) & (pred_bin == 0))
    cm_map = np.zeros(pred_bin.shape, dtype=np.uint8)
    tn_mask = (mask_bin == 0) & (pred_bin == 0)
    tp_mask = (mask_bin == 1) & (pred_bin == 1)
    fn_mask = (mask_bin == 1) & (pred_bin == 0)
    fp_mask = (mask_bin == 0) & (pred_bin == 1)
    cm_map[tn_mask] = 0
    cm_map[tp_mask] = 1
    cm_map[fn_mask] = 2
    cm_map[fp_mask] = 3
    return cm_map, tp, fp, fn

def plot_example(idx, img_file, rgb, mask_np, preds, _, plot_dir):
    from matplotlib.colors import ListedColormap
    ncols = 2 + len(preds)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f'Dataset RGB: {img_file}')
    axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title(f'Dataset Mask\n(Positive pixels: {np.sum(mask_np > 0.5)})')
    axes[1].axis('off')
    for i, (model_name, pred_binary, pred_probs) in enumerate(preds):
        cm_map, tp, fp, fn = get_confusion_map(mask_np, pred_binary)
        cm_colors = ListedColormap([
            [0,0,0,1],    # TN - black
            [1,1,1,1],    # TP - white
            [0,0,1,1],    # FN - blue
            [1,0,0,1]     # FP - red
        ])
        axes[2+i].imshow(cm_map, cmap=cm_colors, vmin=0, vmax=3)
        axes[2+i].set_title(f'{model_name} (.pt)\nTP={tp}, FP={fp}, FN={fn}')
        axes[2+i].axis('off')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/example{idx}_comparison_{img_file}.png', bbox_inches='tight')
    plt.close()

def plot_tiff_validation(idx, tiff_file, tiff_rgb, tiff_preds, plot_dir):
    from matplotlib.colors import ListedColormap
    ncols = 1 + len(tiff_preds)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    axes[0].imshow(tiff_rgb*2)
    axes[0].set_title(f'Validation RGB: {tiff_file}')
    axes[0].axis('off')
    for i, (model_name, pred_binary, pred_probs, _) in enumerate(tiff_preds):
        pred_bin = (pred_binary > 0.5).astype(np.uint8)
        cm_map = np.zeros(pred_bin.shape, dtype=np.uint8)
        cm_map[pred_bin == 0] = 0
        cm_map[pred_bin == 1] = 1
        cm_colors = ListedColormap([
            [0,0,0,1],    # 0 - black
            [1,1,1,1],    # 1 - white
        ])
        axes[1+i].imshow(cm_map, cmap=cm_colors, alpha=0.5, vmin=0, vmax=1)
        axes[1+i].set_title(f'{model_name} (Validation pred)')
        axes[1+i].axis('off')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/validation_{idx}_{tiff_file}.png', bbox_inches='tight')
    plt.close()

def load_s2_tiff_as_tensor(tiff_path):
    arr = tifffile.imread(tiff_path)
    # Remove band 9 if needed (to match training)
    if arr.shape[-1] == 13:
        arr = arr[..., [i for i in range(13) if i != 10]]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

    # normalize between 0 and 1
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    # Scale to match preprocessing

    # tensor *= 15  # Match preprocessing
    return tensor

def normalize_rgb_minmax(rgb):
    rgb = np.clip(rgb, 0, None)
    return np.stack([(band - band.min()) / (band.max() - band.min() + 1e-8) for band in rgb.transpose(2,0,1)], axis=0).transpose(1,2,0)


def main():
    import os
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_dir = 'src/data/processed_unique/images'
    mask_dir = 'src/data/processed_unique/masks'
    downloaded_tiff_path = 'src/google_earth_engine/downloaded_s2_annual_composites'

    def get_image_files(image_dir):
        return sorted([f for f in os.listdir(image_dir) if f.endswith('.pt')])

    def get_mask_files(mask_dir):
        return sorted([f for f in os.listdir(mask_dir) if f.endswith('.pt')])

    def get_tiff_files(tiff_dir):
        return sorted([f for f in os.listdir(tiff_dir) if f.endswith('_multispectral.tif')])

    image_files = get_image_files(image_dir)
    mask_files = get_mask_files(mask_dir)
    tiffs = get_tiff_files(downloaded_tiff_path)
    print(f"Found {len(tiffs)} TIFF files in {downloaded_tiff_path}")

    def load_example_images(image_files, mask_files, image_dir, mask_dir, num_examples=2):
        examples = []
        for img_file, mask_file in zip(image_files, mask_files):
            img = torch.load(os.path.join(image_dir, img_file))
            mask = torch.load(os.path.join(mask_dir, mask_file))
            if mask.sum() > 0:
                examples.append((img, mask, img_file))
            if len(examples) == num_examples:
                break
        return examples

    examples = load_example_images(image_files, mask_files, image_dir, mask_dir)
    if len(examples) < 2:
        print("Not enough test examples with nonzero ground truth masks found.")
        return

    checkpoints = [
        ('unet', 'results/unet_2_best_unet_model.pth'),
        ('convnextv2_open', 'results/convnextv2_open_best_unet_model.pth'),
        ('convnextv2_locked', 'results/convnextv2_locked_best_unet_model.pth')
    ]
    plot_dir = 'inference_plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Process image files with ground truth masks
    for idx, (img, mask, img_file) in enumerate(examples):
        def get_rgb(img):
            if img.shape[0] == 12:
                rgb = img[[3,2,1]].cpu().numpy().transpose(1,2,0)
            else:
                rgb = img[:,:,[3,2,1]].cpu().numpy()
            return normalize_rgb_minmax(rgb)

        rgb = get_rgb(img)
        preds = []
        for model_name, ckpt_path in checkpoints:
            model = get_model(model_name, ckpt_path, device)
            pred_binary, pred_probs = run_inference(model, img, device)
            preds.append((model_name, pred_binary, pred_probs))
        mask_np = mask.squeeze().cpu().numpy()
        plot_example(idx, img_file, rgb, mask_np, preds, [], plot_dir)

    # Process TIFF files (no ground truth, just predictions)
    for idx, tiff_file in enumerate(tiffs):
        tiff_path = os.path.join(downloaded_tiff_path, tiff_file)
        tiff_tensor = load_s2_tiff_as_tensor(tiff_path)
        tiff_rgb = tiff_tensor[[3,2,1]].cpu().numpy().transpose(1,2,0)
        tiff_rgb = normalize_rgb_minmax(tiff_rgb)
        tiff_preds = []
        for model_name, ckpt_path in checkpoints:
            model = get_model(model_name, ckpt_path, device)
            pred_binary, pred_probs = run_inference(model, tiff_tensor, device)
            tiff_preds.append((model_name, pred_binary, pred_probs, tiff_rgb))
        plot_tiff_validation(idx, tiff_file, tiff_rgb, tiff_preds, plot_dir)


if __name__ == '__main__':
    main()