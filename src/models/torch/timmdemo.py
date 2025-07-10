import timm
import torch
from PIL import Image
import requests
from torchvision import transforms

# Download a sample image
img_url = 'https://github.com/pytorch/hub/raw/master/images/dog.jpg'
img = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Preprocessing for timm models
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(img).unsqueeze(0)

# Load a pretrained model from timm
model = timm.create_model('resnet18', pretrained=True)
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Download ImageNet class labels
labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
labels = requests.get(labels_url).text.strip().split('\n')

print('Top-5 predictions:')
for i in range(top5_prob.size(0)):
    print(f"{labels[top5_catid[i]]}: {top5_prob[i].item():.4f}")
