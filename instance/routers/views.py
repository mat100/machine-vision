"""
View router for HTML pages in Machine Vision Instance.
"""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import logging
from pathlib import Path
from ..dependencies import get_config_manager, get_dataset_manager, get_model_manager
from ..config import ConfigManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["views"])

# Setup templates
templates = Jinja2Templates(directory="instance/ui/templates")


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager)
):
    """Main dashboard page."""
    config = config_service.get_config()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request, 
            "config": config,
            "classes": config.classes
        }
    )


@router.get("/data-collection", response_class=HTMLResponse)
async def data_collection_page(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager)
):
    """Data collection page."""
    config = config_service.get_config()
    return templates.TemplateResponse(
        "data_collection.html",
        {
            "request": request, 
            "config": config,
            "classes": config.classes
        }
    )


@router.get("/sorting", response_class=HTMLResponse)
async def sorting_page(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager),
    dataset_manager = Depends(get_dataset_manager)
):
    """Image sorting page."""
    config = config_service.get_config()
    images = dataset_manager.get_all_images()
    return templates.TemplateResponse(
        "sorting.html",
        {
            "request": request, 
            "config": config,
            "classes": config.classes,
            "images": images
        }
    )


@router.get("/training", response_class=HTMLResponse)
async def training_page(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager),
    model_manager = Depends(get_model_manager)
):
    """Training page."""
    config = config_service.get_config()
    models = model_manager.get_models()
    training_status = model_manager.get_training_status()
    return templates.TemplateResponse(
        "training.html",
        {
            "request": request, 
            "config": config,
            "models": models,
            "training_status": training_status
        }
    )


@router.get("/production", response_class=HTMLResponse)
async def production_page(
    request: Request,
    config_service: ConfigManager = Depends(get_config_manager)
):
    """Production page."""
    config = config_service.get_config()
    return templates.TemplateResponse(
        "production.html",
        {
            "request": request, 
            "config": config,
            "classes": config.classes
        }
    )