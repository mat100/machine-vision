"""
Dependency injection for the Machine Vision Hub.
"""
from functools import lru_cache
from .camera.camera_manager import CameraManager
from .registry.instance_manager import InstanceManager
from .config import HubConfig


@lru_cache()
def get_config() -> HubConfig:
    """Get hub configuration singleton."""
    return HubConfig()


@lru_cache()
def get_camera_manager() -> CameraManager:
    """Get camera manager singleton."""
    config = get_config()
    return CameraManager(config)


@lru_cache()
def get_instance_manager() -> InstanceManager:
    """Get instance manager singleton."""
    config = get_config()
    camera_manager = get_camera_manager()
    return InstanceManager(config, camera_manager=camera_manager) 