"""
Data collection API router for Machine Vision Instance.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from typing import Optional
import logging
from ..api.dataset_manager import DatasetManager
from ..api.camera_manager import CameraClient
from ..config import ConfigManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["data-collection"])


from ..dependencies import get_config_manager, get_dataset_manager, get_camera_manager


@router.post("/capture")
async def capture_image(
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
    camera_client: CameraClient = Depends(get_camera_manager)
):
    """Capture image from camera."""
    
    try:
        # Get camera ID from config
        camera_id = config_service.get("camera_id")
        if camera_id is None:
            raise HTTPException(status_code=400, detail="No camera configured")
        
        # Add image from camera directly to dataset
        image_path = await dataset_manager.add_image_from_camera(camera_id)
        if image_path:
            # Return the format that frontend expects
            return {
                "success": True, 
                "path": image_path,
                "image_path": image_path  # Frontend expects this field
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to capture image")
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Upload image file."""
    
    try:
        # Add uploaded file to dataset
        image_path = dataset_manager.add_image_from_file(file)
        if image_path:
            return {"success": True, "path": image_path}
        else:
            raise HTTPException(status_code=400, detail="Failed to save uploaded image")
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-image")
async def classify_image(
    image_path: str = Form(...),
    class_name: str = Form(...),
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Classify an image with a specific class."""
    
    try:
        success = dataset_manager.classify_image(image_path, class_name)
        if success:
            # Return the new image path
            from pathlib import Path
            filename = Path(image_path).name
            new_path = f"classified/{class_name}/{filename}"
            return {"success": True, "new_image_path": new_path}
        else:
            raise HTTPException(status_code=400, detail="Failed to classify image")
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-stats")
async def get_dataset_stats(
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Get dataset statistics."""
    
    try:
        stats = dataset_manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-images")
async def get_recent_images(
    limit: int = 12,
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Get recent images."""
    
    try:
        images = dataset_manager.get_all_images()
        # Limit the results
        recent_images = images[:limit]
        return {"images": recent_images}
    except Exception as e:
        logger.error(f"Error getting recent images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/find-image")
async def find_image(
    filename: str,
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Find image by filename."""
    
    try:
        image_path = dataset_manager.find_image_by_filename(filename)
        if image_path:
            return {"image_path": image_path}
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"Error finding image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-image")
async def delete_image(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager: DatasetManager = Depends(get_dataset_manager)
):
    """Delete an image."""
    
    try:
        data = await request.json()
        image_path = data.get("image_path")
        
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        
        success = dataset_manager.delete_image(image_path)
        if success:
            return {"success": True}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete image")
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 