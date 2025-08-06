"""
Production API router for Machine Vision Instance.
"""
from fastapi import APIRouter, HTTPException, Form, Request, Depends
from fastapi.responses import HTMLResponse
import logging
from datetime import datetime
from pathlib import Path
from ..api.model_manager import ModelManager
from ..api.models import ClassificationResult
from ..config import ConfigManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/production", tags=["production"])


from ..dependencies import get_config_manager, get_model_manager, get_dataset_manager, get_camera_manager


@router.post("/capture-and-classify")
async def production_capture_and_classify(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager),
    dataset_manager = Depends(get_dataset_manager),
    camera_client = Depends(get_camera_manager)
):
    """Capture and classify image in production mode (saves based on settings)."""
    try:
        camera_id = config_service.get("camera_id")
        if camera_id is None:
            raise HTTPException(status_code=400, detail="No camera assigned to this instance")
        
        if not model_manager.active_model:
            raise HTTPException(status_code=400, detail="No active model for classification")
        
        # Check if we should save images based on production settings
        save_images = getattr(model_manager.production_settings, 'save_images', True)
        
        if save_images:
            # Save image to unclassified and then classify
            image_path = await dataset_manager.add_image_from_camera(camera_id)
            if not image_path:
                raise HTTPException(status_code=400, detail="Failed to capture image")
            
            # Classify the saved image
            result = model_manager.classify_image(image_path)
            if result:
                # Update production statistics
                model_manager.production_stats.total_processed += 1
                model_manager.production_stats.successful_classifications += 1
                model_manager.production_stats.last_processed = datetime.now()
                
                # Update confidence distribution
                if result.confidence >= 0.9:
                    model_manager.production_stats.confidence_distribution["high"] += 1
                elif result.confidence >= 0.7:
                    model_manager.production_stats.confidence_distribution["medium"] += 1
                else:
                    model_manager.production_stats.confidence_distribution["low"] += 1
                
                return {
                    "success": True, 
                    "result": result,
                    "image_saved": True,
                    "image_path": image_path,
                    "mode": "production_with_save"
                }
            else:
                raise HTTPException(status_code=400, detail="Classification failed")
        else:
            # Capture directly to memory without saving
            image_bytes = await camera_client.take_snapshot_bytes(camera_id)
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Failed to capture image")
            
            # Classify from memory
            result = model_manager.classify_image_from_memory(image_bytes)
            if result:
                # Update production statistics
                model_manager.production_stats.total_processed += 1
                model_manager.production_stats.successful_classifications += 1
                model_manager.production_stats.last_processed = datetime.now()
                
                # Update confidence distribution
                if result.confidence >= 0.9:
                    model_manager.production_stats.confidence_distribution["high"] += 1
                elif result.confidence >= 0.7:
                    model_manager.production_stats.confidence_distribution["medium"] += 1
                else:
                    model_manager.production_stats.confidence_distribution["low"] += 1
                
                return {
                    "success": True, 
                    "result": result,
                    "image_saved": False,
                    "mode": "production_memory_only"
                }
            else:
                raise HTTPException(status_code=400, detail="Classification failed")
                
    except Exception as e:
        logger.error(f"Error in production capture and classify: {e}")
        raise HTTPException(status_code=500, detail=f"Production error: {str(e)}")


@router.post("/classify-camera-memory")
async def classify_camera_image_memory(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager),
    camera_client = Depends(get_camera_manager)
):
    """Classify image from camera directly from memory without saving to disk."""
    try:
        camera_id = config_service.get("camera_id", 0)
        
        # Capture image from camera as bytes
        image_bytes = await camera_client.take_snapshot_bytes(camera_id)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Failed to capture image")
        
        # Classify the image from memory
        result = model_manager.classify_image_from_memory(image_bytes)
        if result:
            # If production mode is running, update statistics without saving image
            if model_manager.production_mode:
                # Update statistics directly without adding to queue
                model_manager.production_stats.total_processed += 1
                model_manager.production_stats.successful_classifications += 1
                model_manager.production_stats.last_processed = datetime.now()
                
                # Update confidence distribution
                if result.confidence >= 0.9:
                    model_manager.production_stats.confidence_distribution["high"] += 1
                elif result.confidence >= 0.7:
                    model_manager.production_stats.confidence_distribution["medium"] += 1
                else:
                    model_manager.production_stats.confidence_distribution["low"] += 1
            else:
                # If not in production mode, save the result only if save_results is enabled
                if getattr(model_manager.production_settings, 'save_results', True):
                    model_manager._save_production_result(result, 0.0)  # We don't have processing time here
            
            return result
        else:
            raise HTTPException(status_code=400, detail="Classification failed")
    except Exception as e:
        logger.error(f"Error in classify_camera_image_memory: {e}")
        raise HTTPException(status_code=500, detail=f"Camera classification error: {str(e)}")


@router.post("/classify", response_model=ClassificationResult)
async def classify_image_production(
    image_path: str = Form(...),
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Classify image in production mode."""
    
    try:
        # Check if production mode is enabled
        if not config_service.get("production_mode", False):
            raise HTTPException(status_code=400, detail="Production mode not enabled")
        
        # Get active model
        active_model = config_service.get("active_model")
        if not active_model:
            raise HTTPException(status_code=400, detail="No active model")
        
        # Classify image
        result = model_manager.classify_image(image_path, active_model)
        if result:
            return result
        else:
            raise HTTPException(status_code=400, detail="Failed to classify image")
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_production_status(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get production mode status and statistics."""
    try:
        status = model_manager.get_production_status()
        # Add instance state information
        status["instance_state"] = {
            "active_model": config_service.get("active_model"),
            "production_mode": config_service.get("production_mode", False),
            "instance_name": config_service.get("name")
        }
        return status
    except Exception as e:
        logger.error(f"Error getting production status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_production_mode(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Start production mode."""
    
    try:
        success = model_manager.start_production_mode()
        if success:
            # Save production mode state to config
            config_service.set("production_mode", True)
            return {"success": True, "message": "Production mode started"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start production mode")
    except Exception as e:
        logger.error(f"Error starting production mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_production_mode(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Stop production mode."""
    
    try:
        success = model_manager.stop_production_mode()
        if success:
            # Save production mode state to config
            config_service.set("production_mode", False)
            return {"success": True, "message": "Production mode stopped"}
        else:
            raise HTTPException(status_code=400, detail="Failed to stop production mode")
    except Exception as e:
        logger.error(f"Error stopping production mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue")
async def add_to_production_queue(
    image_path: str = Form(...),
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Add an image to the production processing queue."""
    try:
        success = model_manager.add_to_production_queue(image_path)
        if success:
            return {"success": True, "message": "Image added to production queue"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add image to production queue")
    except Exception as e:
        logger.error(f"Error adding to production queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings")
async def update_production_settings(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Update production mode settings."""
    try:
        settings = await request.json()
        success = model_manager.update_production_settings(settings)
        if success:
            # Save settings to config file
            config_service.set("production_settings", settings)
            return {"success": True, "message": "Production settings updated"}
        else:
            raise HTTPException(status_code=400, detail="Failed to update production settings")
    except Exception as e:
        logger.error(f"Error updating production settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-stats")
async def reset_production_stats(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Reset production statistics."""
    try:
        success = model_manager.reset_production_stats()
        if success:
            return {"success": True, "message": "Production statistics reset"}
        else:
            raise HTTPException(status_code=400, detail="Failed to reset production statistics")
    except Exception as e:
        logger.error(f"Error resetting production stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_production_results(
    limit: int = 100,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get recent production results."""
    try:
        results = model_manager.get_production_results(limit)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error getting production results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-stats")
async def update_production_stats(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Manually update production statistics."""
    try:
        # Save current statistics
        model_manager._save_production_stats()
        return {"success": True, "message": "Statistics updated"}
    except Exception as e:
        logger.error(f"Error updating production stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update statistics: {str(e)}")


@router.post("/reload-config")
async def reload_configuration(
    config_service: ConfigManager = Depends(get_config_manager)
):
    """Reload configuration from disk."""
    try:
        config_service.reload_config()
        return {"success": True, "message": "Configuration reloaded"}
    except Exception as e:
        logger.error(f"Error reloading configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")


@router.post("/classify-camera")
async def classify_camera_image(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager),
    dataset_manager = Depends(get_dataset_manager)
):
    """Classify image from camera in production mode."""
    try:
        # Use camera ID 0 if not configured
        camera_id = config_service.get("camera_id", 0)
        
        # Capture image from camera
        image_path = await dataset_manager.add_image_from_camera(camera_id)
        if not image_path:
            raise HTTPException(status_code=400, detail="Failed to capture image")
        
        # Classify the image
        result = model_manager.classify_image(image_path)
        if result:
            # If production mode is running, add to queue for statistics
            if model_manager.production_mode:
                model_manager.add_to_production_queue(image_path)
            else:
                # If not in production mode, save the result only if save_results is enabled
                if getattr(model_manager.production_settings, 'save_results', True):
                    model_manager._save_production_result(result, 0.0)  # We don't have processing time here
            
            return result
        else:
            raise HTTPException(status_code=400, detail="Classification failed")
    except Exception as e:
        logger.error(f"Error in classify_camera_image: {e}")
        raise HTTPException(status_code=500, detail=f"Camera classification error: {str(e)}") 