import tensorflow as tf
import torch
import os
import gzip
from tqdm import tqdm
root = os.path.dirname(__file__)

# Feature schema in tfrecord dataset (B10 not in dataset):
feature_schema = {
    'B2': tf.io.FixedLenFeature([65536], tf.float32),
    'B7': tf.io.FixedLenFeature([65536], tf.float32),
    'solar_panel': tf.io.FixedLenFeature([65536], tf.float32),
    'B5': tf.io.FixedLenFeature([65536], tf.float32),
    'B12': tf.io.FixedLenFeature([65536], tf.float32),
    'B11': tf.io.FixedLenFeature([65536], tf.float32),
    'B6': tf.io.FixedLenFeature([65536], tf.float32),
    # 'B10': tf.io.FixedLenFeature([65536], tf.float32),
    'B9': tf.io.FixedLenFeature([65536], tf.float32),
    'B1': tf.io.FixedLenFeature([65536], tf.float32),
    'B8A': tf.io.FixedLenFeature([65536], tf.float32),
    'B3': tf.io.FixedLenFeature([65536], tf.float32),
    'B4': tf.io.FixedLenFeature([65536], tf.float32),
    'B8': tf.io.FixedLenFeature([65536], tf.float32),
}

# Define the order of Sentinel-2 bands for consistent processing and plotting
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
IMAGE_SIZE = 256 # Each band and the mask is 256x256 pixels, so 256*256 = 65536
NUM_BANDS = len(S2_BANDS)
PLOTS_PER_ENTRY = NUM_BANDS + 1 # All 12 bands + 1 solar_panel mask (not B10).

def parse_example(serialized_example):
    '''
    TensorFlow sample reader.
    Converts datapoint to 
    '''
    parsed_features = tf.io.parse_single_example(serialized_example, feature_schema)

    image_bands = []
    for band_name in S2_BANDS:
        image_bands.append(tf.reshape(parsed_features[band_name], (IMAGE_SIZE, IMAGE_SIZE)))

    image = tf.stack(image_bands, axis=-1) # Resulting shape: (256, 256, 13)
    mask = tf.reshape(parsed_features['solar_panel'], (IMAGE_SIZE, IMAGE_SIZE)) # Resulting shape: (256, 256)
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return image.numpy(), mask.numpy()

def convert_tfrecord_to_pt(tfrecord_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP")

    for idx, raw_example in enumerate(tqdm(dataset)):
        image_np, mask_np = parse_example(raw_example)
        sample = {
            "image": torch.tensor(image_np, dtype=torch.float32) / 255.0,
            "mask": torch.tensor(mask_np, dtype=torch.float32).reshape(1,256,256) / 255.0
        }
        save_path = os.path.join(output_path, f"{idx:05d}.pt")
        torch.save(sample, save_path)

    print(f"Saved {idx+1} samples to {output_path}")

# Example usage:
if __name__ == "__main__":
    # path="src/data/gz_data/solar_2022_global.tfrecord.gz"
    input_path = "src/data/gz_data/solar_2022_global.tfrecord.gz"
    output_path = "src/data/processed"
    convert_tfrecord_to_pt(input_path, output_path)
