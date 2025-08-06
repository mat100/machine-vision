"""
Pydantic models for Machine Vision Instance API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3 = "mobilenet_v3"


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    IDLE = "idle"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class ClassificationResult(BaseModel):
    """Classification result model."""
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Classification timestamp")
    image_path: Optional[str] = Field(None, description="Path to the classified image")
    active_model: Optional[str] = Field(None, description="Active model used for classification")


class TrainingConfig(BaseModel):
    """Training configuration model."""
    architecture: ModelArchitecture = Field(ModelArchitecture.RESNET18, description="Model architecture")
    epochs: int = Field(50, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")
    data_augmentation: bool = Field(True, description="Enable data augmentation")
    early_stopping: bool = Field(True, description="Enable early stopping")
    patience: int = Field(10, ge=1, le=50, description="Early stopping patience")


class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    architecture: ModelArchitecture = Field(..., description="Model architecture")
    created_at: datetime = Field(..., description="Model creation timestamp")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    loss: Optional[float] = Field(None, description="Model loss")
    file_size: Optional[int] = Field(None, description="Model file size in bytes")
    is_active: bool = Field(False, description="Whether this model is currently active")


class DatasetStats(BaseModel):
    """Dataset statistics model."""
    total_images: int = Field(..., description="Total number of images")
    classified_images: int = Field(..., description="Number of classified images")
    unclassified_images: int = Field(..., description="Number of unclassified images")
    class_distribution: Dict[str, int] = Field(..., description="Number of images per class")
    dataset_size_mb: float = Field(..., description="Total dataset size in MB")


class TrainingProgress(BaseModel):
    """Training progress model."""
    status: TrainingStatus = Field(..., description="Current training status")
    current_epoch: int = Field(0, description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    current_loss: Optional[float] = Field(None, description="Current training loss")
    current_accuracy: Optional[float] = Field(None, description="Current training accuracy")
    validation_loss: Optional[float] = Field(None, description="Current validation loss")
    validation_accuracy: Optional[float] = Field(None, description="Current validation accuracy")
    start_time: Optional[datetime] = Field(None, description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    message: Optional[str] = Field(None, description="Status message")


class ImageInfo(BaseModel):
    """Image information model."""
    path: str = Field(..., description="Image file path")
    filename: str = Field(..., description="Image filename")
    class_name: Optional[str] = Field(None, description="Assigned class name")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Image creation timestamp")
    width: Optional[int] = Field(None, description="Image width")
    height: Optional[int] = Field(None, description="Image height") 