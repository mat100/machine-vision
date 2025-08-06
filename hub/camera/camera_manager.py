"""
Camera manager for detecting and managing USB cameras.
"""
import cv2
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from ..config import CameraConfig, CameraStatus, HubConfig


logger = logging.getLogger(__name__)


class CameraManager:
    """Manages USB cameras and their operations."""
    
    def __init__(self, config: HubConfig):
        """Initialize camera manager.
        
        Args:
            config: Hub configuration
        """
        self.config = config
        self.snapshot_dir = config.snapshots_dir
        self.config_file = config.camera_config_file
        self.cameras: Dict[int, CameraConfig] = {}
        self._load_config()
        self._discover_cameras()
    
    def _load_config(self) -> None:
        """Load camera configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    for camera_id, config in config_data.items():
                        self.cameras[int(camera_id)] = CameraConfig(**config)
                logger.info(f"Loaded camera configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading camera configuration: {e}")
    
    def _save_config(self) -> None:
        """Save camera configurations to file."""
        try:
            config_data = {}
            for camera_id, camera in self.cameras.items():
                config_data[str(camera_id)] = camera.model_dump()
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Saved camera configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving camera configuration: {e}")
    
    def _discover_cameras(self) -> None:
        """Discover available USB cameras."""
        logger.info("Discovering USB cameras...")
        
        # Store existing configurations
        existing_configs = {cam_id: cam for cam_id, cam in self.cameras.items()}
        self.cameras.clear()
        
        # Try to open cameras from index 0 to 9
        for camera_id in range(10):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Check if we have existing configuration for this camera
                    if camera_id in existing_configs:
                        existing_config = existing_configs[camera_id]
                        camera_config = CameraConfig(
                            id=camera_id,
                            name=existing_config.name,
                            resolution=existing_config.resolution,
                            fps=existing_config.fps,
                            format=existing_config.format,
                            status=existing_config.status,
                            assigned_to=existing_config.assigned_to
                        )
                        logger.info(f"Using existing configuration for camera {camera_id}")
                    else:
                        camera_config = CameraConfig(
                            id=camera_id,
                            name=f"Camera {camera_id}",
                            resolution=f"{width}x{height}",
                            fps=fps,
                            format="MJPG"
                        )
                        logger.info(f"Found new camera {camera_id}: {camera_config.name}")
                    
                    self.cameras[camera_id] = camera_config
                    cap.release()
                else:
                    break  # No more cameras found
            except Exception as e:
                logger.warning(f"Error checking camera {camera_id}: {e}")
                break
        
        # Save updated configuration
        self._save_config()
        logger.info(f"Discovered {len(self.cameras)} cameras")
    
    def get_cameras(self) -> List[CameraConfig]:
        """Get list of all cameras.
        
        Returns:
            List of camera configurations
        """
        return list(self.cameras.values())
    
    def get_camera(self, camera_id: int) -> Optional[CameraConfig]:
        """Get specific camera configuration.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Camera configuration or None if not found
        """
        return self.cameras.get(camera_id)
    
    def configure_camera(self, camera_id: int, resolution: Optional[str] = None, 
                        fps: Optional[int] = None, format: Optional[str] = None) -> bool:
        """Configure camera parameters.
        
        Args:
            camera_id: Camera ID
            resolution: Resolution string (e.g., "1920x1080")
            fps: Frame rate
            format: Video format (e.g., "MJPG", "YUYV")
            
        Returns:
            True if successful, False otherwise
        """
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # Set resolution if provided
            if resolution:
                try:
                    width, height = map(int, resolution.split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.cameras[camera_id].resolution = resolution
                    logger.info(f"Set camera {camera_id} resolution to {resolution}")
                except ValueError:
                    logger.error(f"Invalid resolution format: {resolution}")
                    cap.release()
                    return False
            
            # Set FPS if provided
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)
                self.cameras[camera_id].fps = fps
                logger.info(f"Set camera {camera_id} FPS to {fps}")
            
            # Set format if provided
            if format:
                # Note: Setting format is more complex and depends on camera capabilities
                # For now, we'll just store it
                self.cameras[camera_id].format = format
                logger.info(f"Set camera {camera_id} format to {format}")
            
            cap.release()
            
            # Save configuration after successful update
            self._save_config()
            return True
            
        except Exception as e:
            logger.error(f"Error configuring camera {camera_id}: {e}")
            return False
    
    def get_camera_capabilities(self, camera_id: int) -> Dict[str, Any]:
        """Get camera capabilities (supported resolutions, FPS, formats).
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dictionary with camera capabilities
        """
        if camera_id not in self.cameras:
            return {}
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return {}
            
            capabilities = {
                "resolutions": [],
                "fps_options": [],
                "formats": []
            }
            
            # Test common resolutions
            common_resolutions = [
                (640, 480), (800, 600), (1024, 768), (1280, 720), 
                (1920, 1080), (2560, 1440), (3840, 2160)
            ]
            
            for width, height in common_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width == width and actual_height == height:
                    capabilities["resolutions"].append(f"{width}x{height}")
            
            # Test common FPS values
            common_fps = [15, 24, 25, 30, 60]
            for fps in common_fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                if actual_fps == fps:
                    capabilities["fps_options"].append(fps)
            
            # Common formats (this is a simplified approach)
            capabilities["formats"] = ["MJPG", "YUYV", "RGB"]
            
            cap.release()
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting camera capabilities for {camera_id}: {e}")
            return {}
    
    def take_snapshot(self, camera_id: int, filename: Optional[str] = None, target_path: Optional[str] = None) -> Optional[str]:
        """Take a snapshot from the specified camera.
        
        Args:
            camera_id: Camera ID
            filename: Optional filename for the snapshot
            target_path: Optional target path where to save the snapshot (if None, uses default snapshot_dir)
            
        Returns:
            Path to the saved snapshot or None if failed
        """
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return None
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return None
            
            # Apply camera configuration
            camera = self.cameras[camera_id]
            if camera.resolution:
                try:
                    width, height = map(int, camera.resolution.split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                except ValueError:
                    logger.warning(f"Invalid resolution format: {camera.resolution}")
            
            if camera.fps:
                cap.set(cv2.CAP_PROP_FPS, camera.fps)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to capture frame from camera {camera_id}")
                return None
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{camera_id}_{timestamp}.jpg"
            
            # Ensure filename has .jpg extension
            if not filename.endswith('.jpg'):
                filename += '.jpg'
            
            # Determine target path
            if target_path:
                # Use provided target path
                target_dir = Path(target_path)
                target_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = target_dir / filename
            else:
                # Use default snapshot directory
                snapshot_path = self.snapshot_dir / filename
            
            # Save the image
            success = cv2.imwrite(str(snapshot_path), frame)
            if success:
                logger.info(f"Snapshot saved: {snapshot_path}")
                return str(snapshot_path)
            else:
                logger.error(f"Failed to save snapshot: {snapshot_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error taking snapshot from camera {camera_id}: {e}")
            return None
    
    def take_snapshot_bytes(self, camera_id: int) -> Optional[bytes]:
        """Take a snapshot from the specified camera and return it as bytes.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Bytes of the captured image or None if failed
        """
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return None
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return None
            
            # Apply camera configuration
            camera = self.cameras[camera_id]
            if camera.resolution:
                try:
                    width, height = map(int, camera.resolution.split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                except ValueError:
                    logger.warning(f"Invalid resolution format: {camera.resolution}")
            
            if camera.fps:
                cap.set(cv2.CAP_PROP_FPS, camera.fps)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to capture frame from camera {camera_id}")
                return None
            
            # Convert frame to bytes
            image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            return image_bytes
                
        except Exception as e:
            logger.error(f"Error taking snapshot from camera {camera_id}: {e}")
            return None
    
    def update_camera_status(self, camera_id: int, status: CameraStatus) -> bool:
        """Update camera status.
        
        Args:
            camera_id: Camera ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        if camera_id in self.cameras:
            self.cameras[camera_id].status = status
            self._save_config()
            return True
        return False
    
    def assign_camera(self, camera_id: int, instance_name: str) -> bool:
        """Assign camera to an instance (add to list of assigned instances).
        
        Args:
            camera_id: Camera ID
            instance_name: Name of the instance
            
        Returns:
            True if successful, False otherwise
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if instance_name not in camera.assigned_to:
                camera.assigned_to.append(instance_name)
            # Update status to IN_USE if any instances are assigned
            if camera.assigned_to:
                camera.status = CameraStatus.IN_USE
            self._save_config()
            return True
        return False
    
    def release_camera(self, camera_id: int, instance_name: str) -> bool:
        """Release camera from specific instance assignment.
        
        Args:
            camera_id: Camera ID
            instance_name: Name of the instance to remove
            
        Returns:
            True if successful, False otherwise
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if instance_name in camera.assigned_to:
                camera.assigned_to.remove(instance_name)
            # Update status to AVAILABLE if no instances are assigned
            if not camera.assigned_to:
                camera.status = CameraStatus.AVAILABLE
            self._save_config()
            return True
        return False
    
    def release_camera_all(self, camera_id: int) -> bool:
        """Release camera from all instance assignments.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            True if successful, False otherwise
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            camera.assigned_to = []
            camera.status = CameraStatus.AVAILABLE
            self._save_config()
            return True
        return False
    
    def refresh_cameras(self) -> None:
        """Refresh camera list by rediscovering cameras."""
        self._discover_cameras() 