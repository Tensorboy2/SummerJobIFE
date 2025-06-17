import matplotlib.pyplot as plt
import os
import numpy as np
import math
import tifffile 

root = os.path.dirname(__file__)
folder = 'downloaded_s2_images'

dummy_locations = [
        {"lon": 139.3767, "lat": 35.9839},
        {"lon": 71.7583232107515, "lat": 38.310451921845235},
        {"lon": 29.638116168955374, "lat": 36.40422688894361},
    {"lon": 118.47251461553486, "lat": 30.108833377834003},
    {"lon": 120.29968790343396, "lat": -28.89431111370441},
    {"lon": -79.31495138076491, "lat": 42.48761799300101},
    {"lon": 84.997693868105, "lat": 21.204732281641306},
    {"lon": 54.04713906905101, "lat": 49.372306330493025},
    {"lon": 64.02382861448243, "lat": 34.9633291732159},
    {"lon": 138.63609948288163, "lat": -18.558295454625195},
    {"lon": -97.03152541417012, "lat": 59.30947000342317},
    {"lon": -87.00093695169154, "lat": 31.31616911969064}
    ]

num_images = len(dummy_locations)
n_cols = 4
n_rows = math.ceil(num_images / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
axes = axes.flatten()

print(f"Attempting to load and display {num_images} images from {os.path.join(root, folder)}...")

for i, loc in enumerate(dummy_locations):
    ax = axes[i]
    lon, lat = loc['lon'], loc['lat']
    title = f"Lat: {lat:.4f}\nLon: {lon:.4f}"

    # Construct the expected filename for the GeoTIFF
    # Remember the filename format was "{lat:.4f}_{lon:.4f}_multispectral.tif"
    # from the previous `fetch_images_from_json` function.
    image_filename = f"{lat:.4f}_{lon:.4f}_multispectral.tif"
    image_path = os.path.join(root, folder, image_filename)

    try:
        if os.path.exists(image_path):
            # Load the multispectral GeoTIFF using tifffile
            # tifffile.imread typically loads as (height, width, channels)
            multispectral_image = tifffile.imread(image_path)
            
            # Clip and normalize for display
            # Sentinel-2 reflectance values are usually 0-10000, so divide by 10000 to scale to 0-1.
            # Or you can clip to a certain range for better visual contrast.
            rgb_display_image = multispectral_image[:, :, [3, 2, 1]] #/ 10000.0 # B4, B3, B2 and normalize

            # For better contrast, you might want to apply a min/max stretch
            # For example, values between 0 and 0.3 for display.
            rgb_display_image = np.clip(rgb_display_image, 0, 0.3) / 0.3

            print(rgb_display_image.shape)
            ax.imshow(rgb_display_image)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.set_title(f"File Not Found\n{title}", fontsize=10, color='red')
            ax.axis('off')
            ax.text(0.5, 0.5, "Image File Not Found", horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, color='red', fontsize=12)
            print(f"Skipping {title} as file was not found: {image_path}")

    except Exception as e:
        ax.set_title(f"Error Loading/Displaying\n{title}", fontsize=10, color='red')
        ax.axis('off')
        ax.text(0.5, 0.5, f"Error: {e}", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, color='red', fontsize=10, wrap=True)
        print(f"An error occurred for {title}: {e}")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()