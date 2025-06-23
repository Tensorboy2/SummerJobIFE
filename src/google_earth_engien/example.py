import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.color import rgb2gray

# ─── 1. Dummy image setup (pretend "green" and "NIR" bands) ───
# We'll simulate a green channel and NIR channel
image = img_as_float(data.astronaut())  # Just a sample RGB image
green = image[:, :, 2]  # Simulate "Green" band (Band 3)
red = image[:, :, 1]  # Simulate "Green" band (Band 3)
nir = rgb2gray(image)   # Simulate NIR band using grayscale as a placeholder

# ─── 2. Compute NDWI ───
ndwi_g = (green - nir) / (green + nir + 1e-8)
ndwi_r = (red - nir) / (red + nir + 1e-8)

# ─── 3. Thresholding for segmentation ───
threshold = 0.5  # Simple global threshold (tune as needed)
water_mask = np.logical_and(ndwi_g,ndwi_r) > threshold

# ─── 4. Plotting ───
fig, axes = plt.subplots(1, 4, figsize=(12, 5))
axes[0].imshow(image)
axes[0].set_title("Original RGB Image")
axes[1].imshow(np.logical_and(ndwi_g,ndwi_r), cmap='RdYlGn')
axes[1].set_title("NDWI")
axes[2].imshow(water_mask, cmap='gray')
axes[2].set_title(f"Water Mask (NDWI > {threshold})")
axes[3].imshow(green, cmap='Greens')
axes[3].set_title("Simulated Green Band")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
