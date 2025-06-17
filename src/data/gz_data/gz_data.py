import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Your provided schema (features for all Sentinel-2 bands and solar_panel mask)
feature_schema = {
    'B2': tf.io.FixedLenFeature([65536], tf.float32),
    'B7': tf.io.FixedLenFeature([65536], tf.float32),
    'solar_panel': tf.io.FixedLenFeature([65536], tf.float32),
    'B5': tf.io.FixedLenFeature([65536], tf.float32),
    'B12': tf.io.FixedLenFeature([65536], tf.float32),
    'B11': tf.io.FixedLenFeature([65536], tf.float32),
    'B6': tf.io.FixedLenFeature([65536], tf.float32),
    'B10': tf.io.FixedLenFeature([65536], tf.float32),
    'B9': tf.io.FixedLenFeature([65536], tf.float32),
    'B1': tf.io.FixedLenFeature([65536], tf.float32),
    'B8A': tf.io.FixedLenFeature([65536], tf.float32),
    'B3': tf.io.FixedLenFeature([65536], tf.float32),
    'B4': tf.io.FixedLenFeature([65536], tf.float32),
    'B8': tf.io.FixedLenFeature([65536], tf.float32),
}

# Define the order of Sentinel-2 bands for consistent processing and plotting
S2_BANDS = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'
]
IMAGE_SIZE = 256 # Each band and the mask is 256x256 pixels, so 256*256 = 65536
NUM_BANDS = len(S2_BANDS)
PLOTS_PER_ENTRY = NUM_BANDS + 1 # All 13 bands + 1 solar_panel mask

def _parse_tfrecord_example(example_proto):
    """
    Parses a single tf.train.Example proto into image bands and a solar panel mask.
    Each feature is reshaped from a 1D array to a 2D (IMAGE_SIZE, IMAGE_SIZE) array.
    The image bands are then stacked into a (H, W, C) tensor.
    """
    parsed_features = tf.io.parse_single_example(example_proto, feature_schema)

    image_bands = []
    for band_name in S2_BANDS:
        image_bands.append(tf.reshape(parsed_features[band_name], (IMAGE_SIZE, IMAGE_SIZE)))

    image = tf.stack(image_bands, axis=-1) # Resulting shape: (256, 256, 13)
    mask = tf.reshape(parsed_features['solar_panel'], (IMAGE_SIZE, IMAGE_SIZE)) # Resulting shape: (256, 256)

    return image, mask

def display_multiple_tfrecord_entries_with_masks(tfrecord_path, num_entries_to_display=3):
    """
    Reads entries from a TFRecord file (can be gzipped), filters for entries
    that have non-empty solar panel masks, and displays their bands and masks
    as subplots, with each entry forming a row.
    """
    print(f"Searching for {num_entries_to_display} entries with non-empty masks from TFRecord: {tfrecord_path}")

    if not os.path.exists(tfrecord_path):
        print(f"Error: File not found at '{tfrecord_path}'. Please verify the path and ensure the file exists.")
        print("Tip: If your script is in a different directory than the data, adjust the `tfrecord_file_path` accordingly.")
        return

    try:
        raw_dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type='GZIP')

        entries_with_masks = []
        entry_counter = 0
        max_search_attempts = 1000 # Limit the number of entries to check to avoid infinite loop on huge files with few masks

        # Iterate through the dataset to find entries with non-empty masks
        for raw_example in raw_dataset:
            if len(entries_with_masks) >= num_entries_to_display:
                break # Found enough entries
            if entry_counter >= max_search_attempts:
                print(f"Stopped searching after {max_search_attempts} entries. Could not find {num_entries_to_display} entries with non-empty masks.")
                break # Reached search limit

            image_tf, mask_tf = _parse_tfrecord_example(raw_example)
            image_np = image_tf.numpy()
            mask_np = mask_tf.numpy()

            # Check if the mask contains any positive values (i.e., solar panels)
            if np.any(mask_np > 0): # Using np.any(mask_np > 0) is more robust than np.sum(mask_np) for float masks
                entries_with_masks.append((image_np, mask_np))
                print(f"Found entry with mask! (Total found: {len(entries_with_masks)}/{num_entries_to_display})")
            entry_counter += 1

        if not entries_with_masks:
            print("No entries with non-empty masks were found in the specified TFRecord file within the search limit.")
            return

        print(f"Displaying {len(entries_with_masks)} entries with non-empty masks.")

        # --- Plotting Setup ---
        total_rows = len(entries_with_masks)
        total_cols = PLOTS_PER_ENTRY

        fig, axes = plt.subplots(total_rows, total_cols, figsize=(12, 9))

        if total_rows == 1:
            axes = [axes] # Ensure axes is always a list of arrays for consistent indexing

        for entry_idx, (image_np, mask_np) in enumerate(entries_with_masks):
            print(f"Plotting Entry {entry_idx+1}: Image shape {image_np.shape}, Mask shape {mask_np.shape}")
            print(f"Plotting Entry {entry_idx+1}: Mask unique values {np.unique(mask_np)}")

            for plot_idx in range(total_cols):
                ax = axes[entry_idx][plot_idx]

                if plot_idx < NUM_BANDS: # It's a band
                    band_name = S2_BANDS[plot_idx]
                    band_data = image_np[:, :, plot_idx]

                    v_min = np.min(band_data)
                    v_max = np.max(band_data)
                    if v_max == v_min:
                        v_max = v_min + 1e-6

                    ax.imshow(band_data, cmap='viridis', vmin=v_min, vmax=v_max)
                    ax.set_title(band_name, fontsize=8)
                else: # It's the mask
                    ax.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
                    ax.set_title("Solar Panel Mask", fontsize=10, weight='bold')

                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.suptitle(f"TFRecord Samples with Masks ({len(entries_with_masks)} found)", y=0.99, fontsize=16, weight='bold')
        plt.show()

    except tf.errors.OpError as e:
        print(f"TensorFlow error while processing TFRecord: {e}")
        print("This might indicate an issue with the TFRecord file's format or data integrity.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your environment, dependencies, and file permissions.")

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the script is in your project root and the data is in src/data/gz_data/
    tfrecord_file_path = os.path.join(current_script_dir, 'solar_2023_us.tfrecord.gz')

    # Display the first 3 entries found that contain solar panels
    display_multiple_tfrecord_entries_with_masks(tfrecord_file_path, num_entries_to_display=10)