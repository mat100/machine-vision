"""
PyTorch model trainer for machine vision classification.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

from .models import ModelArchitecture, TrainingProgress, TrainingStatus

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset for image classification."""
    
    def __init__(self, dataset_path: str, transform=None):
        """Initialize dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            transform: Image transformations
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load images from classified directories
        classified_dir = self.dataset_path / "classified"
        if classified_dir.exists():
            for class_idx, class_name in enumerate(sorted(classified_dir.iterdir())):
                if class_name.is_dir():
                    self.class_to_idx[class_name.name] = class_idx
                    
                    # Load all images in this class
                    for img_path in class_name.glob("*.jpg"):
                        self.images.append(str(img_path))
                        self.labels.append(class_idx)
                    for img_path in class_name.glob("*.png"):
                        self.images.append(str(img_path))
                        self.labels.append(class_idx)
                    for img_path in class_name.glob("*.jpeg"):
                        self.images.append(str(img_path))
                        self.labels.append(class_idx)
        
        logger.info(f"Loaded {len(self.images)} images from {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_model_architecture(architecture: ModelArchitecture, num_classes: int) -> nn.Module:
    """Get PyTorch model based on architecture.
    
    Args:
        architecture: Model architecture
        num_classes: Number of classes
        
    Returns:
        PyTorch model
    """
    def get_in_features(layer):
        """Safely get in_features from a layer."""
        if hasattr(layer, 'in_features'):
            return layer.in_features
        elif hasattr(layer, 'in_channels'):
            return layer.in_channels
        else:
            # For Sequential layers, try to get the first layer's in_features
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                # Find the first layer that has in_features
                for sublayer in layer:
                    try:
                        return get_in_features(sublayer)
                    except ValueError:
                        continue
                raise ValueError(f"Cannot determine in_features for Sequential layer: {layer}")
            # Skip Dropout layers and other layers without in_features
            elif isinstance(layer, (nn.Dropout, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                raise ValueError(f"Layer {layer} does not have in_features")
            else:
                raise ValueError(f"Cannot determine in_features for layer: {layer}")
    
    if architecture == ModelArchitecture.RESNET18:
        model = models.resnet18(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.fc)
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.RESNET34:
        model = models.resnet34(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.fc)
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.RESNET50:
        model = models.resnet50(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.fc)
        model.fc = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.EFFICIENTNET_B0:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.EFFICIENTNET_B1:
        model = models.efficientnet_b1(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.EFFICIENTNET_B2:
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.MOBILENET_V2:
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Linear(in_features, num_classes)
    elif architecture == ModelArchitecture.MOBILENET_V3:
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class ModelTrainer:
    """PyTorch model trainer."""
    
    def __init__(self, dataset_path: str, models_path: str):
        """Initialize trainer.
        
        Args:
            dataset_path: Path to the dataset
            models_path: Path to save models
        """
        self.dataset_path = Path(dataset_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Check for CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train_model(self, architecture: ModelArchitecture, epochs: int, 
                   learning_rate: float, batch_size: int, 
                   progress_callback=None) -> Dict[str, Any]:
        """Train a model.
        
        Args:
            architecture: Model architecture
            epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
            progress_callback: Callback function for progress updates
            
        Returns:
            Training results
        """
        try:
            # Load dataset
            train_transform, val_transform = get_transforms()
            
            # Create datasets
            full_dataset = ImageDataset(str(self.dataset_path), transform=train_transform)
            
            if len(full_dataset) == 0:
                raise ValueError("No images found in dataset")
            
            # Split dataset (80% train, 20% val)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Create model
            model = get_model_architecture(architecture, len(full_dataset.class_to_idx))
            model = model.to(self.device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Training loop
            best_val_acc = 0.0
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    
                    # Update progress
                    if progress_callback:
                        progress_callback({
                            'epoch': epoch + 1,
                            'total_epochs': epochs,
                            'batch': batch_idx + 1,
                            'total_batches': len(train_loader),
                            'train_loss': loss.item(),
                            'train_acc': train_correct / train_total
                        })
                
                scheduler.step()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                # Calculate metrics
                train_loss_avg = train_loss / len(train_loader)
                train_acc_avg = train_correct / train_total
                val_loss_avg = val_loss / len(val_loader)
                val_acc_avg = val_correct / val_total
                
                # Store history
                training_history['train_loss'].append(train_loss_avg)
                training_history['train_acc'].append(train_acc_avg)
                training_history['val_loss'].append(val_loss_avg)
                training_history['val_acc'].append(val_acc_avg)
                
                logger.info(f'Epoch {epoch+1}/{epochs}: '
                          f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, '
                          f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}')
                
                # Save best model
                if val_acc_avg > best_val_acc:
                    best_val_acc = val_acc_avg
                    best_model_state = model.state_dict().copy()
            
            # Save final model
            model_name = f"model_{architecture.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.models_path / f"{model_name}.pth"
            
            logger.info(f"Saving model to {model_path}")
            
            # Save model state dict
            torch.save({
                'model_state_dict': best_model_state,
                'architecture': architecture.value,
                'num_classes': len(full_dataset.class_to_idx),
                'class_to_idx': full_dataset.class_to_idx,
                'training_history': training_history,
                'best_val_acc': best_val_acc,
                'training_config': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Model file size: {model_path.stat().st_size} bytes")
            
            return {
                'model_name': model_name,
                'model_path': str(model_path),
                'architecture': architecture.value,
                'num_classes': len(full_dataset.class_to_idx),
                'class_to_idx': full_dataset.class_to_idx,
                'best_val_acc': best_val_acc,
                'final_train_acc': train_acc_avg,
                'final_val_acc': val_acc_avg,
                'training_history': training_history
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise


def load_trained_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a trained model.
    
    Args:
        model_path: Path to the model file
        device: Device to load model on
        
    Returns:
        Tuple of (model, metadata)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    architecture = ModelArchitecture(checkpoint['architecture'])
    model = get_model_architecture(architecture, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def classify_image_with_model(model: nn.Module, image_path: str, class_to_idx: Dict[str, int], 
                            device: str = 'cpu') -> Tuple[str, float]:
    """Classify an image using a trained model.
    
    Args:
        model: Trained model
        image_path: Path to the image
        class_to_idx: Class to index mapping
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Load and preprocess image
    _, val_transform = get_transforms()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert index back to class name
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        predicted_class = idx_to_class[predicted_idx.item()]
        confidence_value = confidence.item()
    
    return predicted_class, confidence_value


def classify_image_from_memory(model: nn.Module, image, class_to_idx: Dict[str, int], 
                              device: str = 'cpu') -> Tuple[str, float]:
    """Classify an image from memory (PIL Image, numpy array, or bytes) using a trained model.
    
    Args:
        model: Trained model
        image: PIL Image, numpy array, or bytes
        class_to_idx: Class to index mapping
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Load and preprocess image
    _, val_transform = get_transforms()
    
    # Convert different input types to PIL Image
    if isinstance(image, bytes):
        # Handle bytes data (e.g., from camera capture)
        image = Image.open(io.BytesIO(image)).convert('RGB')
    elif hasattr(image, 'shape'):  # numpy array
        image = Image.fromarray(image).convert('RGB')
    elif not isinstance(image, Image.Image):  # other format
        image = Image.open(image).convert('RGB')
    
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert index back to class name
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        predicted_class = idx_to_class[predicted_idx.item()]
        confidence_value = confidence.item()
    
    return predicted_class, confidence_value 