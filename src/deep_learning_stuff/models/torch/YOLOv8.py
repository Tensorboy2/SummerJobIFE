import torch
from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# Step 0: Load the image properly
img = read_image("japan.jpg")  # decode_image expects tensor input, read_image handles file

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Run inference
with torch.no_grad():
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)

# Step 5: Extract class mask for "dog"
class_to_idx = {cls: idx for idx, cls in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["solarpanel"]]

# Step 6: Display using matplotlib
plt.imshow(mask.cpu(), cmap="gray")
plt.title("Dog Class Activation")
plt.axis("off")
plt.show()
