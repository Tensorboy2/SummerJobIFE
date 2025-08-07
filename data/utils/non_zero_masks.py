"""Module for separating non-zero masks from a dataset."""

import torch
import os

pt_path = 'src/data/processed_unique'

image_path = os.path.join(pt_path, 'images')
mask = os.path.join(pt_path, 'masks')

os.makedirs('src/data/processed_unique_non_zero', exist_ok=True)
os.makedirs('src/data/processed_unique_non_zero/images', exist_ok=True)
os.makedirs('src/data/processed_unique_non_zero/masks', exist_ok=True)


def get_non_zero_masks():
    """Extracts non-zero masks from the dataset."""
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.pt')])
    mask_files = sorted([f for f in os.listdir(mask) if f.endswith('.pt')])

    for img_file, mask_file in zip(image_files, mask_files):
        img = torch.load(os.path.join(image_path, img_file))
        msk = torch.load(os.path.join(mask, mask_file))

        if msk.sum() > 0:
            torch.save(img, os.path.join('src/data/processed_unique_non_zero/images', img_file))
            torch.save(msk, os.path.join('src/data/processed_unique_non_zero/masks', mask_file))

if __name__ == "__main__":
    get_non_zero_masks()
    print("Non-zero masks extracted and saved.")
    print("Images and masks saved to 'src/data/processed_unique_non_zero'.")