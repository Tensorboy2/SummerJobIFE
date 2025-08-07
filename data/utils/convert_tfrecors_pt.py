import tensorflow as tf
import torch
import os
import gzip
import hashlib
from tqdm import tqdm

# === Config ===
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
IMAGE_SIZE = 256
NUM_BANDS = len(S2_BANDS)

YEARS = ['2021', '2022', '2023']
INPUT_FOLDER = "src/data/gz_data"
IMAGE_OUT = "src/data/processed_unique/images"
MASK_OUT = "src/data/processed_unique/masks"

os.makedirs(IMAGE_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# === TensorFlow schema ===
feature_schema = {
    band: tf.io.FixedLenFeature([IMAGE_SIZE * IMAGE_SIZE], tf.float32) for band in S2_BANDS
}
feature_schema['solar_panel'] = tf.io.FixedLenFeature([IMAGE_SIZE * IMAGE_SIZE], tf.float32)

def parse_example(serialized_example):
    parsed_features = tf.io.parse_single_example(serialized_example, feature_schema)

    image_bands = []
    for band_name in S2_BANDS:
        image_bands.append(tf.reshape(parsed_features[band_name], (IMAGE_SIZE, IMAGE_SIZE)))

    image = tf.stack(image_bands, axis=-1)  # (256, 256, 12)
    mask = tf.reshape(parsed_features['solar_panel'], (IMAGE_SIZE, IMAGE_SIZE))  # (256, 256)
    return image.numpy(), mask.numpy()

# === Util: hash tensor ===
def hash_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()

# === Main conversion loop ===
def convert_all_years():
    seen_hashes = set()
    total_saved = 0

    for year in YEARS:
        tfrecord_path = os.path.join(INPUT_FOLDER, f"solar_{year}_global.tfrecord.gz")
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")
        print(f"Processing {year}...")

        for idx, raw_example in enumerate(tqdm(dataset, desc=f"{year}")):
            image_np, mask_np = parse_example(raw_example)

            # Normalize and convert to torch tensors
            image_tensor = torch.tensor(image_np, dtype=torch.float32) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=torch.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE) / 255.0

            # Hash to detect duplicates
            image_hash = hash_tensor(image_tensor)
            if image_hash in seen_hashes:
                continue  # Skip duplicates

            # Save unique sample
            torch.save(image_tensor, os.path.join(IMAGE_OUT, f"{total_saved:05d}.pt"))
            torch.save(mask_tensor, os.path.join(MASK_OUT, f"{total_saved:05d}.pt"))
            seen_hashes.add(image_hash)
            total_saved += 1

        print(f"Finished {year}, total unique samples so far: {total_saved}")

    print(f"All done! Final dataset size: {total_saved} unique samples.")

# === Entry point ===
if __name__ == "__main__":
    convert_all_years()
