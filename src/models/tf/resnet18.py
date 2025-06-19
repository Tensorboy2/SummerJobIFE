import tensorflow as tf
from tensorflow.keras import layers, models

# Optional: Mixed precision if supported
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
except:
    pass  # Safe to skip if no GPU support


# ===========================
# 1. TFRecord Schema & Parser
# ===========================
feature_schema = {
    'B1': tf.io.FixedLenFeature([65536], tf.float32),
    'B2': tf.io.FixedLenFeature([65536], tf.float32),
    'B3': tf.io.FixedLenFeature([65536], tf.float32),
    'B4': tf.io.FixedLenFeature([65536], tf.float32),
    'B5': tf.io.FixedLenFeature([65536], tf.float32),
    'B6': tf.io.FixedLenFeature([65536], tf.float32),
    'B7': tf.io.FixedLenFeature([65536], tf.float32),
    'B8': tf.io.FixedLenFeature([65536], tf.float32),
    'B8A': tf.io.FixedLenFeature([65536], tf.float32),
    'B9': tf.io.FixedLenFeature([65536], tf.float32),
    # 'B10': tf.io.FixedLenFeature([65536], tf.float32),
    'B11': tf.io.FixedLenFeature([65536], tf.float32),
    'B12': tf.io.FixedLenFeature([65536], tf.float32),
    'solar_panel': tf.io.FixedLenFeature([65536], tf.float32),
}

def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_schema)

    # Stack and reshape image
    bands = [parsed[key] for key in feature_schema if key != 'solar_panel']
    image = tf.stack(bands, axis=0)
    image = tf.reshape(image, (13, 256, 256))
    image = tf.transpose(image, [1, 2, 0])  # (256, 256, 13)

    # Optional downsampling to save memory
    image = tf.image.resize(image, [128, 128])  # now (128, 128, 13)

    # Label: 1 if any non-zero pixel in mask
    mask = parsed['solar_panel']
    label = tf.cast(tf.reduce_any(tf.not_equal(mask, 0.0)), tf.int32)
    return image, label

# ===========================
# 2. Dataset Loader
# ===========================
def load_dataset(path, batch_size=8, buffer_size=512):
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)  # helps consistent memory usage
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ===========================
# 3. ResNet18-like Model
# ===========================
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

def build_model(input_shape=(128, 128, 13)):
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

# ===========================
# 4. Training Loop
# ===========================
def main():
    path = "src/deep_learning_stuff/models/tf/solar_2023_us.tfrecord.gz"
    batch_size = 2

    dataset = load_dataset(path, batch_size)

    # === Load Model ===
    model = build_model()
    model.load_weights("src/deep_learning_stuff/models/tf/resnet18_solar_classifier.h5")

    # (Optional) Compile only if you plan to evaluate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # === Evaluate ===
    results = model.evaluate(dataset)
    print(f"Eval results: {results}")


if __name__ == "__main__":
    main()
