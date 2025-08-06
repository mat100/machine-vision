"""
Main FastAPI application for the Machine Vision Hub.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from .routers import cameras, instances, views

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Machine Vision Hub", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files
app.mount("/static", StaticFiles(directory="hub/ui/static"), name="static")

# Include routers
app.include_router(views.router)
app.include_router(cameras.router)
app.include_router(instances.router)


if __name__ == "__main__":
    uvicorn.run(
        "hub.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 