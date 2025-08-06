"""
Dependency injection for Machine Vision Instance.
"""
from functools import lru_cache
from .config import ConfigManager
from .api.dataset_manager import DatasetManager
from .api.model_manager import ModelManager
from .api.camera_manager import CameraClient


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get configuration manager singleton."""
    return ConfigManager()


@lru_cache()
def get_dataset_manager() -> DatasetManager:
    """Get dataset manager singleton."""
    config_service = get_config_manager()
    config = config_service.get_config()
    return DatasetManager(
        dataset_path=config.dataset_path,
        hub_url="http://localhost:8000"  # Default hub URL
    )


@lru_cache()
def get_model_manager() -> ModelManager:
    """Get model manager singleton."""
    config_service = get_config_manager()
    config = config_service.get_config()
    
    return ModelManager(
        models_path=config.models_path,
        dataset_path=config.dataset_path,
        active_model=config.active_model,
        config=config_service._config.model_dump()
    )


@lru_cache()
def get_camera_manager() -> CameraClient:
    """Get camera manager singleton."""
    config_service = get_config_manager()
    return CameraClient(hub_url="http://localhost:8000")  # Default hub URL 