import numpy as np
import torch
import os

pt_path = 'src/data/processed_unique'
npy_path = 'src/data/processed_unique_npy'

# Make sure the output folders exist
os.makedirs(os.path.join(npy_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(npy_path, 'masks'), exist_ok=True)

# Get all .pt files from the images folder
names = sorted([f for f in os.listdir(os.path.join(pt_path, 'images')) if f.endswith(".pt")])

for name in names:
    # Load the .pt file (should be a dict with 'image' and 'mask')
    image = torch.load(os.path.join(pt_path, 'images', name)).numpy()
    mask = torch.load(os.path.join(pt_path, 'masks', name)).numpy()

    # Strip .pt extension and save as .npy
    base_name = os.path.splitext(name)[0]
    np.save(os.path.join(npy_path, 'images', base_name + '.npy'), image)
    np.save(os.path.join(npy_path, 'masks', base_name + '.npy'), mask)

print("Conversion complete.")
