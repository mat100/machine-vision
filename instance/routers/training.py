"""
Training API router for Machine Vision Instance.
"""
from fastapi import APIRouter, HTTPException, Depends
import logging
from ..api.model_manager import ModelManager
from ..api.models import TrainingConfig
from ..config import ConfigManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["training"])


from ..dependencies import get_config_manager, get_model_manager


@router.post("/train")
async def start_training(
    training_config: TrainingConfig,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Start model training."""
    
    try:
        # Start training with the provided configuration
        success = model_manager.start_training(
            architecture=training_config.architecture,
            epochs=training_config.epochs,
            learning_rate=training_config.learning_rate,
            batch_size=training_config.batch_size
        )
        if success:
            return {"success": True, "message": "Training started"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start training")
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-status")
async def get_training_status(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get current training status."""
    
    try:
        status = model_manager.get_training_status()
        return status
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models(
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get all available models."""
    
    try:
        models = model_manager.get_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/activate")
async def activate_model(
    model_name: str,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Activate a specific model."""
    
    try:
        success = model_manager.set_active_model(model_name)
        if success:
            # Update config
            config_service.set("active_model", model_name)
            return {"success": True, "message": f"Model {model_name} activated"}
        else:
            raise HTTPException(status_code=400, detail="Failed to activate model")
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Delete a model."""
    
    try:
        success = model_manager.delete_model(model_name)
        if success:
            # If this was the active model, clear it from config
            if config_service.get("active_model") == model_name:
                config_service.set("active_model", None)
            return {"success": True, "message": f"Model {model_name} deleted"}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete model")
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 