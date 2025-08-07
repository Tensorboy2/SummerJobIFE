import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")

# --- 1. Define Identical Architectures ---

# Input shape: (batch_size, height, width, channels) for TF
# Let's assume a small 3x3 grayscale image for simplicity (1 channel)
INPUT_H, INPUT_W, INPUT_C = 3, 3, 1
# Output of Conv: 2 filters
CONV_FILTERS = 2
CONV_KERNEL_SIZE = 2 # 2x2 kernel
CONV_STRIDES = 1
# Input to Linear: Calculated after Conv layer and flattening
# Output of Linear: 1 output neuron
LINEAR_OUTPUT_FEATURES = 1

# Calculate the output dimensions after Conv2D
# H_out = (H_in - K_h) / S_h + 1
# W_out = (W_in - K_w) / S_w + 1
conv_output_h = (INPUT_H - CONV_KERNEL_SIZE) // CONV_STRIDES + 1
conv_output_w = (INPUT_W - CONV_KERNEL_SIZE) // CONV_STRIDES + 1

# Calculate the number of features after Flattening
# flattened_features = conv_output_h * conv_output_w * CONV_FILTERS
flattened_features = conv_output_h * conv_output_w * CONV_FILTERS
# This will be 2 * 2 * 2 = 8

print(f"Calculated Conv2D Output Shape: ({conv_output_h}, {conv_output_w}, {CONV_FILTERS})")
print(f"Calculated Flattened Features: {flattened_features}")


# --- TensorFlow Keras Model ---
def create_tf_model():
    model = keras.Sequential([
        layers.Conv2D(
            filters=CONV_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            strides=CONV_STRIDES,
            padding='valid',
            activation='relu',
            input_shape=(INPUT_H, INPUT_W, INPUT_C),
            name='conv_layer'
        ),
        layers.Flatten(name='flatten_layer'),
        layers.Dense(
            units=LINEAR_OUTPUT_FEATURES,
            activation='sigmoid',
            name='linear_layer'
        )
    ])
    return model

# --- PyTorch Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=INPUT_C,
            out_channels=CONV_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            stride=CONV_STRIDES,
            padding=0
        )
        self.relu = nn.ReLU()
        
        # Use the calculated flattened_features for PyTorch's linear layer as well
        self.linear_layer = nn.Linear(
            in_features=flattened_features, # Corrected input features
            out_features=LINEAR_OUTPUT_FEATURES
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1) # Flatten: (batch_size, channels * height * width)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        return x

# --- 2. Initialize TF Model with Known Weights ---
tf_model = create_tf_model()
tf_model.build(input_shape=(None, INPUT_H, INPUT_W, INPUT_C)) # Build the model to initialize weights
tf_model.summary() # Check the output shape of flatten_layer in the summary

# Set recognizable weights for TF model
tf_conv_kernel = np.array([
    [[[10., 20.]], [[30., 40.]]], # Filter 1, In_channel 1
    [[[50., 60.]], [[70., 80.]]]  # Filter 2, In_channel 1
], dtype=np.float32) # Shape: (kernel_h, kernel_w, in_channels, out_channels) = (2, 2, 1, 2)
tf_conv_bias = np.array([100., 200.], dtype=np.float32) # Shape: (out_channels,) = (2,)

tf_model.get_layer('conv_layer').set_weights([tf_conv_kernel, tf_conv_bias])

# CORRECTED: tf_linear_kernel now has (flattened_features, LINEAR_OUTPUT_FEATURES) shape
tf_linear_kernel = np.array([
    [1.], [2.], [3.], [4.], # Example weights for 8 input features
    [5.], [6.], [7.], [8.]
], dtype=np.float32).reshape(flattened_features, LINEAR_OUTPUT_FEATURES) # Shape: (8, 1)

tf_linear_bias = np.array([0.5], dtype=np.float32) # Shape: (out_features,) = (1,)

tf_model.get_layer('linear_layer').set_weights([tf_linear_kernel, tf_linear_bias])

print("\n--- TensorFlow Model Weights (Original) ---")
print("TF Conv Kernel:\n", tf_model.get_layer('conv_layer').get_weights()[0].squeeze())
print("TF Conv Bias:\n", tf_model.get_layer('conv_layer').get_weights()[1])
print("TF Linear Kernel:\n", tf_model.get_layer('linear_layer').get_weights()[0].squeeze())
print("TF Linear Bias:\n", tf_model.get_layer('linear_layer').get_weights()[1])


# --- 3. Extract TF Weights ---
tf_conv_weights = tf_model.get_layer('conv_layer').get_weights()
tf_conv_w, tf_conv_b = tf_conv_weights[0], tf_conv_weights[1]

tf_linear_weights = tf_model.get_layer('linear_layer').get_weights()
tf_linear_w, tf_linear_b = tf_linear_weights[0], tf_linear_weights[1]


# --- 4. Load PyTorch Model ---
torch_model = SimpleCNN()
print("\n--- PyTorch Model State Dict (Before Transfer) ---")
print(torch_model.state_dict())

# --- 5. Map and Assign Weights ---
pytorch_state_dict = torch_model.state_dict()

# --- Convolutional Layer Weights ---
conv_w_torch = torch.from_numpy(tf_conv_w).permute(3, 2, 0, 1)
conv_b_torch = torch.from_numpy(tf_conv_b)

pytorch_state_dict['conv_layer.weight'].copy_(conv_w_torch)
pytorch_state_dict['conv_layer.bias'].copy_(conv_b_torch)


# --- Linear Layer Weights ---
# CORRECTED: Use the calculated 'flattened_features' for transposition
linear_w_torch = torch.from_numpy(tf_linear_w).T # Transpose (8,1) -> (1,8)
linear_b_torch = torch.from_numpy(tf_linear_b)

pytorch_state_dict['linear_layer.weight'].copy_(linear_w_torch)
pytorch_state_dict['linear_layer.bias'].copy_(linear_b_torch)

# Load the modified state_dict into the PyTorch model
torch_model.load_state_dict(pytorch_state_dict)

print("\n--- PyTorch Model State Dict (After Transfer) ---")
print("PyTorch Conv Weight:\n", pytorch_state_dict['conv_layer.weight'].squeeze())
print("PyTorch Conv Bias:\n", pytorch_state_dict['conv_layer.bias'])
print("PyTorch Linear Weight:\n", pytorch_state_dict['linear_layer.weight'].squeeze())
print("PyTorch Linear Bias:\n", pytorch_state_dict['linear_layer.bias'])


# --- 6. Verify Transfer ---
print("\n--- Verification ---")

# Sample input: a 3x3 grayscale image
tf_input = np.array([
    [[[1.]], [[2.]], [[3.]]],
    [[[4.]], [[5.]], [[6.]]],
    [[[7.]], [[8.]], [[9.]]]
], dtype=np.float32).reshape(1, INPUT_H, INPUT_W, INPUT_C)

torch_input = torch.from_numpy(tf_input).permute(0, 3, 1, 2) # Permute to PyTorch format

print(f"Sample Input (TF format):\n{tf_input.squeeze()}")

# Get output from TF model
tf_output = tf_model(tf_input).numpy()
print(f"\nTF Model Output: {tf_output}")

# Get output from PyTorch model
torch_model.eval()
with torch.no_grad():
    torch_output = torch_model(torch_input).numpy()
print(f"PyTorch Model Output: {torch_output}")

# Compare outputs
tolerance = 1e-5
if np.allclose(tf_output, torch_output, atol=tolerance):
    print("\nSUCCESS: Outputs from TensorFlow and PyTorch models are (almost) identical!")
    print(f"TF Output: {tf_output}")
    print(f"PyTorch Output: {torch_output}")
    print(f"Absolute difference: {np.abs(tf_output - torch_output).max()}")
else:
    print("\nFAILURE: Outputs differ! Check weight transposition and assignment.")
    print(f"TF Output: {tf_output}")
    print(f"PyTorch Output: {torch_output}")
    print(f"Absolute difference: {np.abs(tf_output - torch_output).max()}")