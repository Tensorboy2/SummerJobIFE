import tensorflow as tf
from tensorflow.keras import layers, models

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
    'B10': tf.io.FixedLenFeature([65536], tf.float32),
    'B11': tf.io.FixedLenFeature([65536], tf.float32),
    'B12': tf.io.FixedLenFeature([65536], tf.float32),
    'solar_panel': tf.io.FixedLenFeature([65536], tf.float32),
}

def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_schema)

    # Stack channels into image: (256, 256, 13)
    bands = [parsed[key] for key in feature_schema if key != 'solar_panel']
    image = tf.stack(bands, axis=0)
    image = tf.reshape(image, (13, 256, 256))
    image = tf.transpose(image, [1, 2, 0])  # (256, 256, 13)

    # Binary label: 1 if any pixel is non-zero
    mask = parsed['solar_panel']
    label = tf.cast(tf.reduce_any(tf.not_equal(mask, 0.0)), tf.int32)

    return image, label

# ===========================
# 2. Dataset Loader
# ===========================
def load_dataset(path, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
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

def build_model(input_shape=(256, 256, 13)):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)

# ===========================
# 4. Training Loop
# ===========================
def main():
    path = "src/deep_learning_stuff/models/tf/solar_2023_us.tfrecord.gz"
    batch_size = 2
    epochs = 10

    dataset = load_dataset(path, batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.fit(dataset, epochs=epochs)
    model.save("resnet18_solar_classifier.h5")

if __name__ == "__main__":
    main()
