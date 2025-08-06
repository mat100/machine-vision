"""
Instance manager for handling machine vision instances.
"""
import subprocess
import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import time
from enum import Enum

from ..config import InstanceConfig, InstanceStatus, HubConfig


logger = logging.getLogger(__name__)


class InstanceManager:
    """Manages machine vision instances."""
    
    def __init__(self, config: HubConfig, camera_manager=None, template_dir: str = "instance"):
        """Initialize instance manager.
        
        Args:
            config: Hub configuration
            camera_manager: Camera manager for camera assignments
            template_dir: Directory containing instance template
        """
        self.config = config
        self.instances_dir = config.instances_dir
        self.template_dir = Path(template_dir)
        self.camera_manager = camera_manager
        self.instances: Dict[str, InstanceConfig] = {}
        self._load_instances()
    
    def _load_instances(self) -> None:
        """Load existing instances from disk."""
        logger.info("Loading existing instances...")
        
        for instance_dir in self.instances_dir.iterdir():
            if instance_dir.is_dir():
                config_file = instance_dir / "config.yaml"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        # Remove status from config data - it should not be persisted
                        # Status will be set to default STOPPED and managed in memory
                        config_data.pop('status', None)
                        
                        instance_config = InstanceConfig(**config_data)
                        self.instances[instance_config.name] = instance_config
                        logger.info(f"Loaded instance: {instance_config.name}")
                    except Exception as e:
                        logger.error(f"Error loading instance from {config_file}: {e}")
        
        logger.info(f"Loaded {len(self.instances)} instances")
    
    def create_instance(self, name: str, port: int, camera_id: Optional[int] = None, 
                       classes: Optional[List[str]] = None) -> Optional[InstanceConfig]:
        """Create a new machine vision instance.
        
        Args:
            name: Instance name
            port: Port for the instance API
            camera_id: Optional camera ID to assign
            classes: List of classification classes
            
        Returns:
            Instance configuration or None if failed
        """
        if name in self.instances:
            logger.error(f"Instance {name} already exists")
            return None
        
        # Check if port is available
        if self._is_port_in_use(port):
            logger.error(f"Port {port} is already in use")
            return None
        
        try:
            # Create instance directory
            instance_dir = self.instances_dir / name
            instance_dir.mkdir(exist_ok=True)
            
            # Copy template files
            self._copy_template_files(instance_dir)
            
            # Create configuration
            instance_config = InstanceConfig(
                name=name,
                port=port,
                camera_id=camera_id,
                classes=classes or ["OK", "NG"],
                dataset_path=str(instance_dir / "dataset"),
                models_path=str(instance_dir / "models"),
                config_path=str(instance_dir / "config.yaml")
            )
            
            # Save configuration
            self._save_instance_config(instance_config)
            
            # Assign camera if specified
            if camera_id is not None and self.camera_manager:
                self.camera_manager.assign_camera(camera_id, name)
                logger.info(f"Assigned camera {camera_id} to instance {name}")
            
            # Add to instances dict
            self.instances[name] = instance_config
            
            logger.info(f"Created instance: {name}")
            return instance_config
            
        except Exception as e:
            logger.error(f"Error creating instance {name}: {e}")
            return None
    
    def delete_instance(self, name: str) -> bool:
        """Delete an instance.
        
        Args:
            name: Instance name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.instances:
            logger.error(f"Instance {name} not found")
            return False
        
        try:
            # Stop instance if running
            self.stop_instance(name)
            
            # Release camera assignment if any
            if self.camera_manager:
                instance = self.instances[name]
                if instance.camera_id is not None:
                    self.camera_manager.release_camera(instance.camera_id, name)
                    logger.info(f"Released camera {instance.camera_id} from instance {name}")
            
            # Remove instance directory
            instance_dir = self.instances_dir / name
            if instance_dir.exists():
                shutil.rmtree(instance_dir)
            
            # Remove from instances dict
            del self.instances[name]
            
            logger.info(f"Deleted instance: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting instance {name}: {e}")
            return False
    
    def start_instance(self, name: str) -> bool:
        """Start an instance.
        
        Args:
            name: Instance name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.instances:
            logger.error(f"Instance {name} not found")
            return False
        
        instance = self.instances[name]
        
        if instance.status == InstanceStatus.RUNNING:
            logger.warning(f"Instance {name} is already running")
            return True
        
        try:
            # Update status to starting
            instance.status = InstanceStatus.STARTING
            
            # Start instance process from main project directory
            # Instance data is in ~/.machine-vision/instances/{name}/
            # but code runs from the main project directory
            project_root = Path(__file__).parent.parent.parent  # Go up to project root
            cmd = [
                "python", "-m", "uvicorn", 
                "instance.main:app",
                "--host", "0.0.0.0",
                "--port", str(instance.port),
                "--reload"
            ]
            
            # Start process in background from project root
            # Pass instance name as environment variable
            env = os.environ.copy()
            env['INSTANCE_NAME'] = name
            
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Wait a bit for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                instance.status = InstanceStatus.RUNNING
                logger.info(f"Started instance: {name}")
                return True
            else:
                instance.status = InstanceStatus.ERROR
                logger.error(f"Failed to start instance: {name}")
                return False
                
        except Exception as e:
            instance.status = InstanceStatus.ERROR
            logger.error(f"Error starting instance {name}: {e}")
            return False
    
    def stop_instance(self, name: str) -> bool:
        """Stop an instance.
        
        Args:
            name: Instance name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.instances:
            logger.error(f"Instance {name} not found")
            return False
        
        instance = self.instances[name]
        
        if instance.status == InstanceStatus.STOPPED:
            logger.warning(f"Instance {name} is already stopped")
            return True
        
        try:
            # Find and kill process
            self._kill_instance_process(name, instance.port)
            
            instance.status = InstanceStatus.STOPPED
            logger.info(f"Stopped instance: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping instance {name}: {e}")
            return False
    
    def get_instances(self) -> List[InstanceConfig]:
        """Get list of all instances.
        
        Returns:
            List of instance configurations
        """
        return list(self.instances.values())
    
    def get_instance(self, name: str) -> Optional[InstanceConfig]:
        """Get specific instance configuration.
        
        Args:
            name: Instance name
            
        Returns:
            Instance configuration or None if not found
        """
        return self.instances.get(name)
    
    def _copy_template_files(self, instance_dir: Path) -> None:
        """Create instance directory structure (no code copying needed)."""
        # Instance directory should only contain configuration and data
        # The instance code runs from the main project directory
        logger.info(f"Creating instance directory structure for {instance_dir}")
        
        # Create basic directory structure
        (instance_dir / "dataset").mkdir(parents=True, exist_ok=True)
        (instance_dir / "models").mkdir(parents=True, exist_ok=True)
        (instance_dir / "production").mkdir(parents=True, exist_ok=True)
    
    def _save_instance_config(self, instance_config: InstanceConfig) -> None:
        """Save instance configuration to file."""
        config_file = Path(instance_config.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Pydantic model to dict
        config_dict = instance_config.model_dump()
        
        # Remove status from config - it should not be persisted
        config_dict.pop('status', None)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _kill_instance_process(self, name: str, port: int) -> None:
        """Kill instance process by port."""
        try:
            # Find process using the port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                pid = result.stdout.strip()
                subprocess.run(["kill", "-9", pid])
                logger.info(f"Killed process {pid} for instance {name}")
        except Exception as e:
            logger.warning(f"Could not kill process for instance {name}: {e}") 