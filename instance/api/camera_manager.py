"""
Camera client for communicating with the hub's camera manager.
"""
import logging
import httpx
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraClient:
    """Client for communicating with the hub's camera manager."""
    
    def __init__(self, hub_url: str = "http://localhost:8000"):
        """Initialize camera client.
        
        Args:
            hub_url: URL of the hub application
        """
        self.hub_url = hub_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def take_snapshot(self, camera_id: int, target_path: Optional[str] = None) -> Optional[str]:
        """Take a snapshot from the specified camera.
        
        Args:
            camera_id: Camera ID
            target_path: Optional target path where to save the snapshot
            
        Returns:
            Path to the saved snapshot or None if failed
        """
        try:
            # Prepare form data
            form_data = {}
            if target_path:
                form_data["target_path"] = target_path
            
            response = await self.client.post(
                f"{self.hub_url}/api/cameras/{camera_id}/snapshot",
                data=form_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("path")
            
            logger.error(f"Failed to take snapshot from camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error taking snapshot from camera {camera_id}: {e}")
            return None
    
    async def take_snapshot_bytes(self, camera_id: int) -> Optional[bytes]:
        """Take a snapshot from the specified camera and return as bytes.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Image bytes or None if failed
        """
        try:
            response = await self.client.post(
                f"{self.hub_url}/api/cameras/{camera_id}/snapshot-bytes"
            )
            
            if response.status_code == 200:
                return response.content
            
            logger.error(f"Failed to take snapshot bytes from camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error taking snapshot bytes from camera {camera_id}: {e}")
            return None
    
    async def get_cameras(self):
        """Get list of available cameras."""
        try:
            response = await self.client.get(f"{self.hub_url}/api/cameras")
            if response.status_code == 200:
                return response.json().get("cameras", [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting cameras: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 