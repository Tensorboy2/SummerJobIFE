import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import tensorflow as tf
from tensorflow.keras import layers, models

# ==== Define your ResNet18 model architecture ====
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if downsample or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model(input_shape=(256, 256, 12)):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters, downsample=True)
        x = residual_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # Cast output back to float32

    return tf.keras.Model(inputs, x)

# ==== Load model weights ====
model = build_model()
model.load_weights("src/google_earth_engien/resnet18_solar_classifier.h5")
print("✅ Model loaded successfully.")

# ==== Configuration ====
root = os.path.dirname(__file__)
folder = 'downloaded_s2_images'

dummy_locations = [
    {"lon": 139.3767, "lat": 35.9839},
    {"lon": 6.0155, "lat": 52.4872},
    {"lon": 6.1405, "lat": 52.4844},
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

# ==== Plot setup ====
num_images = len(dummy_locations)
n_cols = 4
n_rows = math.ceil(num_images / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes = axes.flatten()

# ==== Main loop ====
for i, loc in enumerate(dummy_locations):
    ax = axes[i]
    lat, lon = loc['lat'], loc['lon']
    title = f"Lat: {lat:.2f}\nLon: {lon:.2f}"
    image_filename = f"{lat:.4f}_{lon:.4f}_multispectral.tif"
    image_path = os.path.join(root, folder, image_filename)

    try:
        if os.path.exists(image_path):
            img = tifffile.imread(image_path)

            if img.shape != (256, 256, 12):
                raise ValueError(f"Expected shape (256, 256, 12), got {img.shape}")

            # Preprocessing for RGB display (B4, B3, B2 = indices 3, 2, 1)
            rgb_display = img[:, :, [3, 2, 1]]
            rgb_display = np.clip(rgb_display, 0, 0.3) / (0.3)

            # Preprocessing for model
            x = img.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            # Inference
            pred = model.predict(x)[0][0]

            ax.imshow(rgb_display)
            ax.set_title(f"{title}\nPred: {pred:.6f} ", fontsize=9)
            ax.axis('off')

        else:
            ax.set_title(f"File Not Found\n{title}", fontsize=10, color='red')
            ax.axis('off')
            print(f"❌ File not found: {image_path}")

    except Exception as e:
        ax.set_title(f"Error\n{title}", fontsize=10, color='red')
        ax.axis('off')
        print(f"⚠️ Error processing {image_filename}: {e}")

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
