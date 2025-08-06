"""
Camera API router for the Machine Vision Hub.
"""
from fastapi import APIRouter, HTTPException, Form, Depends, Response
from typing import Optional
from ..camera.camera_manager import CameraManager
from ..dependencies import get_camera_manager

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/")
async def get_cameras(camera_manager: CameraManager = Depends(get_camera_manager)):
    """Get all cameras."""
    return {"cameras": camera_manager.get_cameras()}


@router.get("/{camera_id}")
async def get_camera(camera_id: int, camera_manager: CameraManager = Depends(get_camera_manager)):
    """Get specific camera details."""
    camera = camera_manager.get_camera(camera_id)
    if camera:
        return {"camera": camera}
    else:
        raise HTTPException(status_code=404, detail="Camera not found")


@router.get("/{camera_id}/capabilities")
async def get_camera_capabilities(camera_id: int, camera_manager: CameraManager = Depends(get_camera_manager)):
    """Get camera capabilities."""
    capabilities = camera_manager.get_camera_capabilities(camera_id)
    if capabilities:
        return {"capabilities": capabilities}
    else:
        raise HTTPException(status_code=404, detail="Camera not found")


@router.post("/{camera_id}/configure")
async def configure_camera(
    camera_id: int,
    camera_manager: CameraManager = Depends(get_camera_manager),
    resolution: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    format: Optional[str] = Form(None)
):
    """Configure camera parameters."""
    success = camera_manager.configure_camera(
        camera_id=camera_id,
        resolution=resolution,
        fps=fps,
        format=format
    )
    
    if success:
        camera = camera_manager.get_camera(camera_id)
        return {"success": True, "camera": camera.model_dump()}
    else:
        raise HTTPException(status_code=400, detail="Failed to configure camera")


@router.post("/{camera_id}/snapshot")
async def take_snapshot(
    camera_id: int, 
    camera_manager: CameraManager = Depends(get_camera_manager),
    target_path: Optional[str] = Form(None)
):
    """Take a snapshot from camera."""
    snapshot_path = camera_manager.take_snapshot(camera_id, target_path=target_path)
    if snapshot_path:
        return {"success": True, "path": snapshot_path}
    else:
        raise HTTPException(status_code=400, detail="Failed to take snapshot")


@router.post("/{camera_id}/snapshot-bytes")
async def take_snapshot_bytes(
    camera_id: int, 
    camera_manager: CameraManager = Depends(get_camera_manager)
):
    """Take a snapshot from camera and return as bytes."""
    image_bytes = camera_manager.take_snapshot_bytes(camera_id)
    if image_bytes:
        return Response(content=image_bytes, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=400, detail="Failed to take snapshot")


@router.post("/refresh")
async def refresh_cameras(camera_manager: CameraManager = Depends(get_camera_manager)):
    """Refresh camera list."""
    camera_manager.refresh_cameras()
    return {"success": True, "cameras": camera_manager.get_cameras()} 