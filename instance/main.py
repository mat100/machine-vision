"""
Main FastAPI application for Machine Vision Instance.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
import os
from pathlib import Path

# Import routers
from .routers import views, images, data_collection, production, training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Machine Vision Instance",
    description="Machine vision instance for data collection, training, and classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files
app.mount("/static", StaticFiles(directory="instance/ui/static"), name="static")

# Include routers
app.include_router(views.router)
app.include_router(images.router)
app.include_router(data_collection.router)
app.include_router(production.router)
app.include_router(training.router)

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8001))
    
    uvicorn.run(
        "instance.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 