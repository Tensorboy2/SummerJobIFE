'''
gym.py

Module for training torch modules on 
'''
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import os

from trainer import Trainer_cl
from deep_learning_stuff.models.torch.classifier import Classifier

def create_data_transforms():
    """Create data transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(train_dir, val_dir, batch_size=32):
    """Create data loaders from directories"""
    train_transform, val_transform = create_data_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, 
                                 download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model (CIFAR-10 has 10 classes)
    model = Classifier(num_classes=10, input_channels=3)
    
    # Create trainer
    trainer = Trainer_cl(model, train_loader, val_loader, device)
    
    # Train the model
    trainer.train(epochs=5)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save the model
    trainer.save_model('image_classifier.pth')