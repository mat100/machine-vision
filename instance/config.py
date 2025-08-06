"""
Configuration models and service for Machine Vision Instance.
"""
import yaml
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class InstanceStatus(str, Enum):
    """Instance status enumeration."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"


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
    production_settings: Optional[dict] = None
    
    class Config:
        """Pydantic configuration."""
        # Disable protected namespaces to avoid warnings
        protected_namespaces = ()


class ConfigManager:
    """Manager for instance configuration."""
    
    def __init__(self):
        """Initialize configuration service."""
        self.instance_name = os.environ.get('INSTANCE_NAME', 'default-instance')
        self.config_path = Path.home() / ".machine-vision" / "instances" / self.instance_name / "config.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> InstanceConfig:
        """Load instance configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    logger.info(f"Loaded config for instance {self.instance_name}: {config_data}")
                    
                    # Handle legacy model_architecture field
                    if "model_architecture" in config_data:
                        config_data["architecture"] = config_data.pop("model_architecture")
                    
                    # Convert string status back to enum if needed, or use default if corrupted
                    if 'status' in config_data:
                        if isinstance(config_data['status'], str):
                            try:
                                config_data['status'] = InstanceStatus(config_data['status'])
                            except ValueError:
                                logger.warning(f"Invalid status value '{config_data['status']}', using default")
                                config_data['status'] = InstanceStatus.STOPPED
                        else:
                            logger.warning(f"Corrupted status value, using default")
                            config_data['status'] = InstanceStatus.STOPPED
                    
                    # Ensure config_path is set
                    config_data["config_path"] = str(self.config_path)
                    
                    return InstanceConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return self._get_default_config()
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> InstanceConfig:
        """Get default configuration."""
        return InstanceConfig(
            name=self.instance_name,
            port=8001,
            camera_id=None,
            classes=["OK", "NG"],
            architecture="resnet18",  # Changed from model_architecture
            dataset_path=str(Path.home() / ".machine-vision" / "instances" / self.instance_name / "dataset"),
            models_path=str(Path.home() / ".machine-vision" / "instances" / self.instance_name / "models"),
            config_path=str(self.config_path),
            active_model=None,
            production_mode=False,
            production_settings={}
        )
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self._config.model_dump()
            # Remove config_path from saved data as it's internal
            config_dict.pop("config_path", None)
            
            # Convert enum values to strings for YAML serialization
            if 'status' in config_dict and isinstance(config_dict['status'], Enum):
                config_dict['status'] = config_dict['status'].value
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Saved config for instance {self.instance_name}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update_config(self, key: str, value: Any) -> None:
        """Update a specific config value and save."""
        setattr(self._config, key, value)
        self.save_config()
    
    def get_config(self) -> InstanceConfig:
        """Get current configuration."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific config value."""
        return getattr(self._config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value and save."""
        self.update_config(key, value)
    
    def reload_config(self) -> None:
        """Reload configuration from disk."""
        self._config = self._load_config()
        logger.info(f"Reloaded config for instance {self.instance_name}") 