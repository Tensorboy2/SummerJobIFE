import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

class ResNetFeatureExtractor(nn.Module):
    """Extract multi-scale features using ResNet18 for higher resolution segmentation"""
    
    def __init__(self, pretrained=True, feature_layer='layer3'):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained ResNet18
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        # Extract different layers for different resolutions
        self.conv1 = resnet.conv1      # 1/2 resolution, 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4 resolution
        
        self.layer1 = resnet.layer1    # 1/4 resolution, 64 channels
        self.layer2 = resnet.layer2    # 1/8 resolution, 128 channels
        self.layer3 = resnet.layer3    # 1/16 resolution, 256 channels
        self.layer4 = resnet.layer4    # 1/32 resolution, 512 channels
        
        self.feature_layer = feature_layer
        
        # Upsampling layers for higher resolution output
        if feature_layer == 'layer2':
            self.upsample = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
            self.output_channels = 128
        elif feature_layer == 'layer3':
            self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
            self.output_channels = 256
        else:  # layer4
            self.upsample = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.output_channels = 512
    
    def forward(self, x):
        # Forward through ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # 1/4 resolution
        
        if self.feature_layer == 'layer1':
            features = x
        else:
            x = self.layer2(x)  # 1/8 resolution
            if self.feature_layer == 'layer2':
                features = x
            else:
                x = self.layer3(x)  # 1/16 resolution
                if self.feature_layer == 'layer3':
                    features = x
                else:
                    x = self.layer4(x)  # 1/32 resolution
                    features = x
        
        # Upsample for higher resolution
        if hasattr(self, 'upsample'):
            features = self.upsample(features)
        
        return features

def extract_patch_features(feature_extractor, dataloader, device='cuda'):
    """
    Extract higher resolution features from image patches for segmentation
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = images.to(device)
            
            # Extract features using ResNet
            features = feature_extractor(images)  # Higher resolution features
            
            batch_size, channels, h, w = features.shape
            print(f"Feature map size: {h}x{w}, channels: {channels}")
            
            # Reshape features for patch-based processing
            # Each spatial location becomes a sample
            features_flat = features.permute(0, 2, 3, 1)  # (B, H, W, C)
            features_flat = features_flat.contiguous().view(-1, channels)  # (B*H*W, C)
            
            # Create corresponding labels for each spatial location
            # For this example, we'll use the original labels repeated for each patch
            # In practice, you'd have pixel-level ground truth labels
            if len(labels.shape) == 1:  # Classification labels
                # Repeat labels for each spatial location
                patch_labels = labels.unsqueeze(1).unsqueeze(2).repeat(1, h, w)
                patch_labels = patch_labels.view(-1)
            else:  # Already spatial labels
                # Downsample labels to match feature map size
                patch_labels = torch.nn.functional.interpolate(
                    labels.float().unsqueeze(1), 
                    size=(h, w), 
                    mode='nearest'
                ).squeeze(1).long().view(-1)
            
            all_features.append(features_flat.cpu().numpy())
            all_labels.append(patch_labels.cpu().numpy())
            
            # Limit samples for memory efficiency (remove this for full dataset)
            if batch_idx >= 10:  # Process only first 10 batches for demo
                break
    
    return np.vstack(all_features), np.hstack(all_labels)

def train_random_forest_segmentation():
    """Main function to train Random Forest on ResNet features for segmentation"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset (using CIFAR-10 as example)
    # Replace with your segmentation dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize ResNet feature extractor with higher resolution features
    # Use 'layer2' for 4x higher resolution, 'layer3' for 2x higher resolution
    feature_extractor = ResNetFeatureExtractor(pretrained=True, feature_layer='layer2').to(device)
    print(f"Using feature layer: layer2 for higher resolution segmentation")
    
    print("Extracting training features...")
    train_features, train_labels = extract_patch_features(
        feature_extractor, train_loader, device=device
    )
    
    print("Extracting test features...")
    test_features, test_labels = extract_patch_features(
        feature_extractor, test_loader, device=device
    )
    
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    
    # Train Random Forest classifier
    print("Training Random Forest...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_classifier.fit(train_features, train_labels)
    
    # Make predictions
    print("Making predictions...")
    train_predictions = rf_classifier.predict(train_features)
    test_predictions = rf_classifier.predict(test_features)
    
    # Evaluate performance
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    
    # Feature importance
    feature_importance = rf_classifier.feature_importances_
    print(f"\nTop 10 most important features:")
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, feat_idx in enumerate(top_features):
        print(f"{i+1}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
    
    return feature_extractor, rf_classifier

def segment_image(feature_extractor, rf_classifier, image, device='cuda', output_size=None):
    """
    Segment a single image using the trained model with higher resolution output
    """
    feature_extractor.eval()
    original_size = image.shape[-2:]  # Get original image size
    
    with torch.no_grad():
        # Add batch dimension and move to device
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        
        # Extract features
        features = feature_extractor(image)  # Higher resolution features
        
        # Reshape for prediction
        _, channels, h, w = features.shape
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        
        # Predict using Random Forest
        predictions = rf_classifier.predict(features_flat.cpu().numpy())
        
        # Reshape back to spatial dimensions
        segmentation_map = predictions.reshape(h, w)
        
        # Upsample to original image size for higher resolution output
        if output_size is not None:
            segmentation_tensor = torch.from_numpy(segmentation_map).float().unsqueeze(0).unsqueeze(0)
            segmentation_upsampled = torch.nn.functional.interpolate(
                segmentation_tensor,
                size=output_size,
                mode='nearest'
            )
            segmentation_map = segmentation_upsampled.squeeze().numpy().astype(int)
        
        return segmentation_map

class MultiScaleResNetExtractor(nn.Module):
    """Extract and combine multi-scale features for highest resolution segmentation"""
    
    def __init__(self, pretrained=True):
        super(MultiScaleResNetExtractor, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels, 1/4 resolution
        self.layer2 = resnet.layer2  # 128 channels, 1/8 resolution
        self.layer3 = resnet.layer3  # 256 channels, 1/16 resolution
        
        # Upsampling layers to bring all features to 1/4 resolution
        self.upsample_2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.upsample_3 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=4, padding=2)
        
        # Feature fusion
        self.feature_fusion = nn.Conv2d(64 + 128 + 256, 256, kernel_size=1)
        self.output_channels = 256
    
    def forward(self, x):
        # Forward through initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract multi-scale features
        feat1 = self.layer1(x)      # 1/4 resolution, 64 channels
        feat2 = self.layer2(feat1)  # 1/8 resolution, 128 channels
        feat3 = self.layer3(feat2)  # 1/16 resolution, 256 channels
        
        # Upsample higher-level features to 1/4 resolution
        feat2_up = self.upsample_2(feat2)  # 128 channels at 1/4 resolution
        feat3_up = self.upsample_3(feat3)  # 256 channels at 1/4 resolution
        
        # Concatenate all features
        combined_features = torch.cat([feat1, feat2_up, feat3_up], dim=1)
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)
        
        return fused_features

def train_high_resolution_segmentation():
    """Train with multi-scale features for highest resolution segmentation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for memory
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Use multi-scale feature extractor
    feature_extractor = MultiScaleResNetExtractor(pretrained=True).to(device)
    print("Using multi-scale ResNet extractor for highest resolution")
    
    print("Extracting training features...")
    train_features, train_labels = extract_patch_features(
        feature_extractor, train_loader, device=device
    )
    
    print("Extracting test features...")
    test_features, test_labels = extract_patch_features(
        feature_extractor, test_loader, device=device
    )
    
    print(f"Training features shape: {train_features.shape}")
    
    # Train Random Forest with optimized parameters for higher resolution
    print("Training Random Forest for high-resolution segmentation...")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=25,      # Deeper trees for complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_classifier.fit(train_features, train_labels)
    
    # Evaluate
    test_predictions = rf_classifier.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"High-resolution segmentation accuracy: {test_accuracy:.4f}")
    
    return feature_extractor, rf_classifier

def visualize_segmentation(original_image, segmentation_map):
    """Visualize original image and segmentation result"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = original_image * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        original_image = original_image.permute(1, 2, 0).numpy()
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation map
    axes[1].imshow(segmentation_map, cmap='tab10')
    axes[1].set_title('Segmentation Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the model
    feature_extractor, rf_classifier = train_random_forest_segmentation()
    
    # Example: Segment a test image
    # Load a test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    
    # Get a sample image
    sample_image, _ = test_dataset[0]
    
    # Perform segmentation with higher resolution output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmentation_result = segment_image(
        feature_extractor, 
        rf_classifier, 
        sample_image, 
        device, 
        output_size=(224, 224)  # Upsample to original image size
    )
    
    # Visualize results
    visualize_segmentation(sample_image, segmentation_result)
    
    print("Segmentation complete!")