"""
Images router for serving dataset images.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import logging
from ..config import ConfigManager
from ..dependencies import get_config_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])


@router.get("/images/")
async def list_images(
    config_service: ConfigManager = Depends(get_config_manager)
):
    """List available images in the dataset."""
    try:
        config = config_service.get_config()
        dataset_path = Path(config.dataset_path)
        
        images = []
        
        # Check unclassified directory
        unclassified_path = dataset_path / "unclassified"
        if unclassified_path.exists():
            for img_file in unclassified_path.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append({
                        "filename": img_file.name,
                        "path": f"/images/{img_file.name}",
                        "category": "unclassified"
                    })
        
        # Check classified directories
        classified_dir = dataset_path / "classified"
        if classified_dir.exists():
            for class_dir in classified_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.glob("*"):
                        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            images.append({
                                "filename": img_file.name,
                                "path": f"/images/{img_file.name}",
                                "category": class_name
                            })
        
        # Check Hub snapshots directory
        hub_snapshots_path = Path.home() / ".machine-vision" / "snapshots"
        if hub_snapshots_path.exists():
            for img_file in hub_snapshots_path.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append({
                        "filename": img_file.name,
                        "path": f"/images/{img_file.name}",
                        "category": "snapshots"
                    })
        
        return JSONResponse(content={
            "message": "Available images",
            "count": len(images),
            "images": images
        })
        
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/images/{filename}")
async def serve_image(
    filename: str,
    config_service: ConfigManager = Depends(get_config_manager)
):
    """Serve images from dataset."""
    try:
        config = config_service.get_config()
        dataset_path = Path(config.dataset_path)
        
        # Look for image in unclassified and classified directories
        unclassified_path = dataset_path / "unclassified" / filename
        
        if unclassified_path.exists():
            return FileResponse(str(unclassified_path))
        
        # Check in classified directories
        classified_dir = dataset_path / "classified"
        for class_dir in classified_dir.iterdir():
            if class_dir.is_dir():
                image_path = class_dir / filename
                if image_path.exists():
                    return FileResponse(str(image_path))
        
        # If not found in dataset, try Hub snapshots directory
        hub_snapshots_path = Path.home() / ".machine-vision" / "snapshots" / filename
        if hub_snapshots_path.exists():
            return FileResponse(str(hub_snapshots_path))
        
        raise HTTPException(status_code=404, detail="Image not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 