"""
Configuration models for the Machine Vision Hub.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from pathlib import Path
import os


class CameraStatus(str, Enum):
    """Camera status enumeration."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class InstanceStatus(str, Enum):
    """Instance status enumeration."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"


class CameraConfig(BaseModel):
    """Camera configuration model."""
    id: int
    name: str
    status: CameraStatus = CameraStatus.AVAILABLE
    resolution: Optional[str] = None
    fps: Optional[int] = None
    format: Optional[str] = None
    assigned_to: list[str] = []  # List of instance names that can use this camera


class InstanceConfig(BaseModel):
    """Instance configuration model."""
    name: str
    port: int = 8001
    status: InstanceStatus = InstanceStatus.STOPPED
    camera_id: Optional[int] = None
    classes: list[str] = ["OK", "NG"]
    dataset_path: str
    models_path: str
    config_path: Optional[str] = None
    architecture: str = "resnet18"  # Changed from model_architecture
    active_model: Optional[str] = None
    production_mode: bool = False
    
    class Config:
        """Pydantic configuration."""
        # Disable protected namespaces to avoid warnings
        protected_namespaces = ()


class HubConfig(BaseModel):
    """Hub configuration model."""
    # Data directories - use user's home directory
    data_dir: Path = Path.home() / ".machine-vision"
    snapshots_dir: Path = Path.home() / ".machine-vision" / "snapshots"
    instances_dir: Path = Path.home() / ".machine-vision" / "instances"
    camera_config_file: Path = Path.home() / ".machine-vision" / "camera_config.json"
    
    # Application settings
    hub_port: int = 8000
    instance_port_range: tuple[int, int] = (8001, 8999)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create data directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.instances_dir.mkdir(parents=True, exist_ok=True) 