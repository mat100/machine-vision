#!/usr/bin/env python3
"""
Script to run the Machine Vision Hub.
"""
import sys
import os
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Machine Vision Hub...")
    print("Access the web interface at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "hub.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 