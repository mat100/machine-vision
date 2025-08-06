"""
Dataset manager for handling image storage and classification.
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import cv2
from PIL import Image
import json
import asyncio

from .models import DatasetStats, ImageInfo
from .camera_manager import CameraClient


logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset operations including image storage and classification."""
    
    def __init__(self, dataset_path: str, hub_url: str = "http://localhost:8000"):
        """Initialize dataset manager.
        
        Args:
            dataset_path: Path to the dataset directory
            hub_url: URL of the hub application
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.unclassified_dir = self.dataset_path / "unclassified"
        self.classified_dir = self.dataset_path / "classified"
        self.metadata_file = self.dataset_path / "metadata.json"
        
        self.unclassified_dir.mkdir(exist_ok=True)
        self.classified_dir.mkdir(exist_ok=True)
        
        # Initialize camera client
        self.camera_client = CameraClient(hub_url)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Convert classes list back to set
                    if "classes" in metadata:
                        if isinstance(metadata["classes"], list):
                            metadata["classes"] = set(metadata["classes"])
                        elif not isinstance(metadata["classes"], set):
                            metadata["classes"] = set()
                    else:
                        metadata["classes"] = set()
                    return metadata
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        return {
            "images": {},
            "classes": set(),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_metadata(self) -> None:
        """Save dataset metadata."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        # Convert set to list for JSON serialization
        self.metadata["classes"] = list(self.metadata["classes"])
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    async def add_image_from_camera(self, camera_id: int) -> Optional[str]:
        """Add image from camera capture.
        
        Args:
            camera_id: Camera ID to capture from
            
        Returns:
            Path to the saved image or None if failed
        """
        try:
            # Take snapshot directly to unclassified directory
            # Don't generate timestamp here - let the hub camera manager do it
            # to avoid timestamp mismatch
            
            # Take snapshot directly to our dataset directory
            snapshot_path = await self.camera_client.take_snapshot(
                camera_id, 
                target_path=str(self.unclassified_dir)
            )
            
            if not snapshot_path:
                logger.error(f"Failed to take snapshot from camera {camera_id}")
                return None
            
            # The snapshot is already saved to the correct location
            image_path = Path(snapshot_path)
            
            # Use the actual filename from the saved file
            filename = image_path.name
            
            # Add to metadata
            self.metadata["images"][str(image_path)] = {
                "filename": filename,
                "class_name": None,
                "size_bytes": image_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "source": "camera",
                "camera_id": camera_id,
                "original_snapshot": None  # No longer needed since we save directly
            }
            
            self._save_metadata()
            logger.info(f"Added image from camera: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Error adding image from camera: {e}")
            return None
    
    def classify_image(self, image_path: str, class_name: str) -> bool:
        """Classify an image with a class.
        
        Args:
            image_path: Path to the image
            class_name: Class name to assign
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_path = Path(image_path)
            
            # Create class directory
            class_dir = self.classified_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Find the actual file location and current class
            actual_file_path = None
            current_class = None
            current_metadata_key = None
            
            # First, try to find the file by name in metadata
            for metadata_key, info in self.metadata["images"].items():
                metadata_path = Path(metadata_key)
                if metadata_path.name == image_path.name:
                    current_class = info.get("class_name")
                    current_metadata_key = metadata_key
                    if metadata_path.exists():
                        actual_file_path = metadata_path
                        break
            
            # If not found in metadata, check if the provided path exists
            if actual_file_path is None and image_path.exists():
                actual_file_path = image_path
            
            # If still not found, search in filesystem
            if actual_file_path is None:
                # Check unclassified directory
                unclassified_path = self.unclassified_dir / image_path.name
                if unclassified_path.exists():
                    actual_file_path = unclassified_path
                
                # Check classified directories
                if actual_file_path is None:
                    for class_dir_iter in self.classified_dir.iterdir():
                        if class_dir_iter.is_dir():
                            classified_path = class_dir_iter / image_path.name
                            if classified_path.exists():
                                actual_file_path = classified_path
                                current_class = class_dir_iter.name
                                break
            
            # If file not found anywhere, return False
            if actual_file_path is None:
                logger.error(f"Image not found anywhere: {image_path.name}")
                return False
            
            # If image is already classified in the same class, do nothing
            if current_class == class_name:
                logger.info(f"Image {image_path.name} is already classified as {class_name}")
                return True
            
            # Move/copy the file to the new class directory
            new_path = class_dir / image_path.name
            
            # If the file is in a different location, move it
            if str(actual_file_path) != str(new_path):
                shutil.move(str(actual_file_path), str(new_path))
            
            # Update metadata
            # Remove old entry if it exists
            if current_metadata_key and current_metadata_key in self.metadata["images"]:
                del self.metadata["images"][current_metadata_key]
            
            # Add new entry
            self.metadata["images"][str(new_path)] = {
                "filename": image_path.name,
                "class_name": class_name,
                "size_bytes": new_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "source": "classification" if current_class is None else "reclassification"
            }
            
            # Add class to set (ensure it's a set)
            if not isinstance(self.metadata["classes"], set):
                self.metadata["classes"] = set(self.metadata["classes"]) if isinstance(self.metadata["classes"], list) else set()
            self.metadata["classes"].add(class_name)
            
            self._save_metadata()
            logger.info(f"Classified image {image_path.name} as {class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return False
    
    def get_unclassified_images(self) -> List[ImageInfo]:
        """Get list of unclassified images.
        
        Returns:
            List of unclassified image information
        """
        images = []
        for image_path_str, info in self.metadata["images"].items():
            image_path = Path(image_path_str)
            if image_path.exists() and image_path.parent == self.unclassified_dir and info["class_name"] is None:
                images.append(ImageInfo(
                    path=image_path_str,
                    filename=info["filename"],
                    class_name=info["class_name"],
                    size_bytes=info["size_bytes"],
                    created_at=datetime.fromisoformat(info["created_at"])
                ))
        
        return sorted(images, key=lambda x: x.created_at, reverse=True)
    
    def get_all_images(self) -> List[ImageInfo]:
        """Get list of all images (classified and unclassified).
        
        Returns:
            List of all image information
        """
        images = []
        
        # Get all images from metadata
        for image_path_str, info in self.metadata["images"].items():
            image_path = Path(image_path_str)
            if image_path.exists():
                images.append(ImageInfo(
                    path=image_path_str,
                    filename=info["filename"],
                    class_name=info["class_name"],
                    size_bytes=info["size_bytes"],
                    created_at=datetime.fromisoformat(info["created_at"])
                ))
        
        return sorted(images, key=lambda x: x.created_at, reverse=True)
    
    def get_statistics(self) -> DatasetStats:
        """Get dataset statistics.
        
        Returns:
            Dataset statistics
        """
        total_images = len(self.metadata["images"])
        classified_images = sum(1 for info in self.metadata["images"].values() 
                              if info["class_name"] is not None)
        unclassified_images = total_images - classified_images
        
        # Calculate class distribution
        class_distribution = {}
        for info in self.metadata["images"].values():
            if info["class_name"]:
                class_distribution[info["class_name"]] = class_distribution.get(info["class_name"], 0) + 1
        
        # Calculate total size
        total_size_bytes = sum(info["size_bytes"] for info in self.metadata["images"].values())
        dataset_size_mb = total_size_bytes / (1024 * 1024)
        
        return DatasetStats(
            total_images=total_images,
            classified_images=classified_images,
            unclassified_images=unclassified_images,
            class_distribution=class_distribution,
            dataset_size_mb=dataset_size_mb
        )
    
    def find_image_by_filename(self, filename: str) -> Optional[str]:
        """Find image by filename in dataset.
        
        Args:
            filename: Name of the image file to find
            
        Returns:
            Full path to the image if found, None otherwise
        """
        try:
            # Search in metadata for the filename
            for image_path_str, info in self.metadata["images"].items():
                if info["filename"] == filename:
                    image_path = Path(image_path_str)
                    if image_path.exists():
                        return str(image_path)
            
            # If not found in metadata, search in filesystem
            # Check unclassified directory
            unclassified_path = self.unclassified_dir / filename
            if unclassified_path.exists():
                return str(unclassified_path)
            
            # Check classified directories
            for class_dir in self.classified_dir.iterdir():
                if class_dir.is_dir():
                    classified_path = class_dir / filename
                    if classified_path.exists():
                        return str(classified_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding image by filename: {e}")
            return None

    def delete_image(self, image_path: str) -> bool:
        """Delete an image from the dataset.
        
        Args:
            image_path: Path to the image to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_path = Path(image_path)
            if image_path.exists():
                image_path.unlink()
            
            # Remove from metadata
            if str(image_path) in self.metadata["images"]:
                del self.metadata["images"][str(image_path)]
            
            self._save_metadata()
            logger.info(f"Deleted image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False 