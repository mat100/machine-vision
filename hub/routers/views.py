"""
View router for HTML pages in the Machine Vision Hub.
"""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ..camera.camera_manager import CameraManager
from ..registry.instance_manager import InstanceManager
from ..dependencies import get_camera_manager, get_instance_manager

router = APIRouter(tags=["views"])

# Setup templates
templates = Jinja2Templates(directory="hub/ui/templates")


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    camera_manager: CameraManager = Depends(get_camera_manager),
    instance_manager: InstanceManager = Depends(get_instance_manager)
):
    """Main dashboard page."""
    cameras = camera_manager.get_cameras()
    instances = instance_manager.get_instances()
    
    # Convert Pydantic models to dictionaries for JSON serialization
    cameras_dict = [camera.model_dump() for camera in cameras]
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "cameras": cameras_dict,
            "instances": instances
        }
    )


@router.get("/cameras", response_class=HTMLResponse)
async def cameras_page(
    request: Request,
    camera_manager: CameraManager = Depends(get_camera_manager)
):
    """Cameras management page."""
    cameras = camera_manager.get_cameras()
    
    return templates.TemplateResponse(
        "cameras.html",
        {
            "request": request,
            "cameras": cameras
        }
    )


@router.get("/instances", response_class=HTMLResponse)
async def instances_page(
    request: Request,
    camera_manager: CameraManager = Depends(get_camera_manager),
    instance_manager: InstanceManager = Depends(get_instance_manager)
):
    """Instances management page."""
    instances = instance_manager.get_instances()
    cameras = camera_manager.get_cameras()
    
    return templates.TemplateResponse(
        "instances.html",
        {
            "request": request,
            "instances": instances,
            "cameras": cameras
        }
    ) 