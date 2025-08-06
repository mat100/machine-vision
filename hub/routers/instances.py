"""
Instance API router for the Machine Vision Hub.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from ..registry.instance_manager import InstanceManager
from ..camera.camera_manager import CameraManager
from ..dependencies import get_camera_manager, get_instance_manager

router = APIRouter(prefix="/api/instances", tags=["instances"])


@router.get("/")
async def get_instances(instance_manager: InstanceManager = Depends(get_instance_manager)):
    """Get all instances."""
    instances = instance_manager.get_instances()
    # Convert Pydantic models to dictionaries for JSON serialization
    instances_dict = [instance.model_dump() for instance in instances]
    return {"instances": instances_dict}


class CreateInstanceRequest(BaseModel):
    name: str
    port: int
    camera_id: Optional[int] = None


@router.post("/")
async def create_instance(
    request: CreateInstanceRequest,
    camera_manager: CameraManager = Depends(get_camera_manager),
    instance_manager: InstanceManager = Depends(get_instance_manager)
):
    """Create a new instance."""
    instance = instance_manager.create_instance(
        name=request.name,
        port=request.port,
        camera_id=request.camera_id,
        classes=["OK", "NG"]  # Default classes
    )
    
    if instance:
        # Assign camera if specified
        if request.camera_id is not None:
            camera_manager.assign_camera(request.camera_id, request.name)
        
        return {"success": True, "instance": instance.model_dump()}
    else:
        raise HTTPException(status_code=400, detail="Failed to create instance")


@router.post("/{name}/start")
async def start_instance(name: str, instance_manager: InstanceManager = Depends(get_instance_manager)):
    """Start an instance."""
    success = instance_manager.start_instance(name)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Failed to start instance")


@router.post("/{name}/stop")
async def stop_instance(name: str, instance_manager: InstanceManager = Depends(get_instance_manager)):
    """Stop an instance."""
    success = instance_manager.stop_instance(name)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Failed to stop instance")


@router.delete("/{name}")
async def delete_instance(
    name: str,
    camera_manager: CameraManager = Depends(get_camera_manager),
    instance_manager: InstanceManager = Depends(get_instance_manager)
):
    """Delete an instance."""
    # Release camera if assigned
    instance = instance_manager.get_instance(name)
    if instance and instance.camera_id is not None:
        camera_manager.release_camera(instance.camera_id)
    
    success = instance_manager.delete_instance(name)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Failed to delete instance") 