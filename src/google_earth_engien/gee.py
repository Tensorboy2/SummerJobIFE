import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile

def get_mask(tiff, bands):
    '''
    Returns a binary mask for likely water-based solar panels:
    1. Detect water using MNDWI
    2. Remove vegetation using NDVI
    3. [Optional] Restrict based on brightness/flatness
    '''
    # Extract bands
    blue = tiff[:, :, bands['blue']]
    green = tiff[:, :, bands['green']]
    red = tiff[:, :, bands['red']]
    nir = tiff[:, :, bands['nir']]
    swir_1 = tiff[:, :, bands['swir_1']]
    swir_2 = tiff[:, :, bands['swir_2']]
    # swir_2 not used here

    BI_green = (blue - green) / (blue + green + 1e-6) > -0.07
    BI_red   = (blue - red) / (blue + red + 1e-6) > -0.05
    NDBI     = (swir_1 - nir) / (swir_1 + nir + 1e-6) > 0.02
    NSDSI    = (swir_1 - swir_2) / (swir_1 + swir_2 + 1e-6) > 0.12

    mask = (BI_green.astype(int) &
            # (swir_1.astype(int)>0.07) &
        BI_red.astype(int) &
        NDBI.astype(int) &
        NSDSI.astype(int))

    return mask

# ==== Configuration ====
root = os.path.dirname(__file__)
folder = 'downloaded_s2_annual_composites'

dummy_locations = [
    {"lon": 139.3767, "lat": 35.9839},
    {"lon": 6.0155, "lat": 52.4872},
    {"lon": 6.1405, "lat": 52.4844},
    {"lon": 118.47251461553486, "lat": 30.108833377834003},
]
bands = {
    'blue': 1, 'green': 2, 'red': 3, 'nir': 7, 'swir_1': 10, 'swir_2':11 
}


# ==== Plot setup ====
num_images = len(dummy_locations)
n_cols = 4
n_rows = math.ceil(num_images / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes = axes.flatten()
years = ['2021','2022','2023']

# ==== Main loop ====
for i, loc in enumerate(dummy_locations):
    ax = axes[i]
    lat, lon = loc['lat'], loc['lon']
    title = f"Lat: {lat:.2f}\nLon: {lon:.2f}"

    # Store yearly masks
    yearly_masks = []

    for year in years:
        image_filename = f"{lat:.4f}_{lon:.4f}_{year}_multispectral.tif"
        image_path = os.path.join(root, folder, image_filename)

        if os.path.exists(image_path):
            img = tifffile.imread(image_path)

            if img.shape != (256, 256, 12):
                raise ValueError(f"Expected shape (256, 256, 12), got {img.shape}")

            mask = get_mask(img, bands)
            yearly_masks.append(mask)

    if not yearly_masks:
        ax.set_title(f"{title}\n(No Data)", fontsize=8)
        ax.axis('off')
        continue

    # === Combine yearly masks ===
    mask_stack = np.stack(yearly_masks, axis=0)  # Shape: (num_years, H, W)
    mask_sum = mask_stack.mean(axis=0)            # Count of active years per pixel

    # Pick a threshold (e.g., 3 out of 5 years)
    consistency_threshold = 0.3
    consistent_mask = mask_sum >= consistency_threshold

    # Load any image for visualization (e.g., the latest year)
    img = tifffile.imread(os.path.join(root, folder, f"{lat:.4f}_{lon:.4f}_{years[-1]}_multispectral.tif"))
    rgb_display = img[:, :, [3, 2, 1]]
    rgb_display = np.clip(rgb_display, 0, 0.3) / (0.3)

    overlay = rgb_display.copy()
    overlay[consistent_mask] = [1.0, 0.0, 0.0]  # Red highlight

    ax.imshow(overlay)
    ax.set_title(f"{title}\nConsistent ({consistency_threshold}+ years)", fontsize=8)
    ax.axis('off')


# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
