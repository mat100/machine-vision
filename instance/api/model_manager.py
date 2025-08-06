"""
Model manager for handling ML model training and inference.
"""
import os
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import threading
import time
import queue
from dataclasses import dataclass, asdict

from .models import ModelInfo, TrainingProgress, TrainingStatus, ClassificationResult, ModelArchitecture
from .trainer import ModelTrainer, load_trained_model, classify_image_with_model, classify_image_from_memory


logger = logging.getLogger(__name__)


@dataclass
class ProductionSettings:
    """Production mode settings."""
    batch_size: int = 1
    confidence_threshold: float = 0.8
    max_processing_time: float = 5.0  # seconds
    enable_logging: bool = True
    save_results: bool = True
    save_images: bool = True  # Whether to save captured images to unclassified folder
    results_path: str = "production_results"
    auto_retry_failed: bool = True
    max_retries: int = 3


@dataclass
class ProductionStats:
    """Production statistics."""
    total_processed: int = 0
    successful_classifications: int = 0
    failed_classifications: int = 0
    average_processing_time: float = 0.0
    start_time: Optional[datetime] = None
    last_processed: Optional[datetime] = None
    confidence_distribution: Dict[str, int] = None  # "high", "medium", "low"
    
    def __post_init__(self):
        if self.confidence_distribution is None:
            self.confidence_distribution = {"high": 0, "medium": 0, "low": 0}


class ModelManager:
    """Manages ML model operations including training and inference."""
    
    def __init__(self, models_path: str, dataset_path: str = None, active_model: str = None, config: dict = None):
        """Initialize model manager.
        
        Args:
            models_path: Path to the models directory
            dataset_path: Path to the dataset directory
            active_model: Name of the active model to load
            config: Configuration dictionary
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.config = config or {}
        
        self.metadata_file = self.models_path / "metadata.json"
        self.active_model: Optional[str] = None
        
        # Training state
        self.training_thread: Optional[threading.Thread] = None
        self.training_progress = TrainingProgress(
            status=TrainingStatus.IDLE,
            total_epochs=0
        )
        
        # Production mode state
        self.production_mode = False
        self.production_settings = ProductionSettings()
        self.production_stats = ProductionStats()
        self.production_queue = queue.Queue()
        self.production_thread: Optional[threading.Thread] = None
        
        # Create production directory (sibling to models directory)
        self.production_path = self.models_path.parent / "production"
        self.production_path.mkdir(exist_ok=True)
        self.production_results_path = self.production_path / "results"
        self.production_results_path.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load production settings from config
        self._load_production_settings()
        
        # Load production statistics if they exist
        self._load_production_stats()
        
        # Set active model if provided
        if active_model:
            self.set_active_model(active_model)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load models metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading models metadata: {e}")
        
        return {
            "models": {},
            "active_model": None,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_metadata(self) -> None:
        """Save models metadata."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["active_model"] = self.active_model
        
        try:
            logger.info(f"Saving metadata to {self.metadata_file}")
            logger.info(f"Metadata contains {len(self.metadata['models'])} models: {list(self.metadata['models'].keys())}")
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving models metadata: {e}")
    
    def _load_production_settings(self) -> None:
        """Load production settings from config."""
        # Get production settings from config, use empty dict if None
        production_settings = self.config.get('production_settings') or {}
        
        # Create default ProductionSettings instance
        default_settings = ProductionSettings()
        
        # Update production settings with values from config, fallback to defaults
        self.production_settings.confidence_threshold = production_settings.get('confidence_threshold', default_settings.confidence_threshold)
        self.production_settings.max_processing_time = production_settings.get('max_processing_time', default_settings.max_processing_time)
        self.production_settings.save_results = production_settings.get('save_results', default_settings.save_results)
        self.production_settings.save_images = production_settings.get('save_images', default_settings.save_images)
        self.production_settings.enable_logging = production_settings.get('enable_logging', default_settings.enable_logging)
        self.production_settings.batch_size = production_settings.get('batch_size', default_settings.batch_size)
        self.production_settings.auto_retry_failed = production_settings.get('auto_retry_failed', default_settings.auto_retry_failed)
        self.production_settings.max_retries = production_settings.get('max_retries', default_settings.max_retries)
        self.production_settings.results_path = production_settings.get('results_path', default_settings.results_path)
        
        logger.info("Loaded production settings from config")
    
    def _save_production_settings(self) -> None:
        """Save production settings to config."""
        # Update config with production settings
        if 'production_settings' not in self.config:
            self.config['production_settings'] = {}
        
        self.config['production_settings']['confidence_threshold'] = self.production_settings.confidence_threshold
        self.config['production_settings']['max_processing_time'] = self.production_settings.max_processing_time
        self.config['production_settings']['save_results'] = self.production_settings.save_results
        self.config['production_settings']['save_images'] = self.production_settings.save_images
        self.config['production_settings']['enable_logging'] = self.production_settings.enable_logging
        self.config['production_settings']['batch_size'] = self.production_settings.batch_size
        self.config['production_settings']['auto_retry_failed'] = self.production_settings.auto_retry_failed
        self.config['production_settings']['max_retries'] = self.production_settings.max_retries
        
        logger.info("Updated production settings in config")
    
    def _load_production_stats(self) -> None:
        """Load production statistics from file."""
        stats_file = self.production_path / "production_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                    # Convert string timestamps back to datetime
                    if stats_data.get('start_time'):
                        stats_data['start_time'] = datetime.fromisoformat(stats_data['start_time'])
                    if stats_data.get('last_processed'):
                        stats_data['last_processed'] = datetime.fromisoformat(stats_data['last_processed'])
                    self.production_stats = ProductionStats(**stats_data)
                logger.info("Loaded production statistics")
            except Exception as e:
                logger.error(f"Error loading production statistics: {e}")
    
    def _save_production_stats(self) -> None:
        """Save production statistics to file."""
        stats_file = self.production_path / "production_stats.json"
        try:
            stats_dict = asdict(self.production_stats)
            # Convert datetime objects to strings for JSON serialization
            if stats_dict.get('start_time'):
                stats_dict['start_time'] = stats_dict['start_time'].isoformat()
            if stats_dict.get('last_processed'):
                stats_dict['last_processed'] = stats_dict['last_processed'].isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            logger.info("Saved production statistics")
        except Exception as e:
            logger.error(f"Error saving production statistics: {e}")
    
    def get_models(self) -> List[ModelInfo]:
        """Get list of trained models.
        
        Returns:
            List of model information
        """
        # Reload metadata to get latest changes
        self.metadata = self._load_metadata()
        
        logger.info(f"Getting models. Metadata contains {len(self.metadata['models'])} models")
        logger.info(f"Models in metadata: {list(self.metadata['models'].keys())}")
        
        models = []
        for model_name, info in self.metadata["models"].items():
            model_path = self.models_path / f"{model_name}.pth"
            file_size = model_path.stat().st_size if model_path.exists() else None
            
            logger.info(f"Processing model: {model_name}, file exists: {model_path.exists()}")
            
            models.append(ModelInfo(
                name=model_name,
                version=info["version"],
                architecture=ModelArchitecture(info["architecture"]),
                created_at=datetime.fromisoformat(info["created_at"]),
                accuracy=info.get("accuracy"),
                loss=info.get("loss"),
                file_size=file_size,
                is_active=(model_name == self.active_model)
            ))
        
        logger.info(f"Returning {len(models)} models")
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def start_training(self, architecture: ModelArchitecture, epochs: int, 
                      learning_rate: float, batch_size: int) -> bool:
        """Start model training.
        
        Args:
            architecture: Model architecture
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            True if training started successfully, False otherwise
        """
        if self.training_progress.status == TrainingStatus.TRAINING:
            logger.warning("Training already in progress")
            return False
        
        try:
            # Update training progress
            self.training_progress = TrainingProgress(
                status=TrainingStatus.TRAINING,
                current_epoch=0,
                total_epochs=epochs,
                start_time=datetime.now()
            )
            
            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._train_model,
                args=(architecture, epochs, learning_rate, batch_size)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            logger.info(f"Started training with {architecture} for {epochs} epochs")
            return True
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            self.training_progress.status = TrainingStatus.FAILED
            return False
    
    def _train_model(self, architecture: ModelArchitecture, epochs: int, 
                    learning_rate: float, batch_size: int) -> None:
        """Train model in background thread."""
        try:
            logger.info("Starting model training...")
            
            # Get dataset path
            if not self.dataset_path:
                raise ValueError("Dataset path not configured")
            
            # Initialize trainer
            trainer = ModelTrainer(str(self.dataset_path), str(self.models_path))
            
            # Progress callback function
            def progress_callback(progress_data):
                if self.training_progress.status == TrainingStatus.TRAINING:
                    self.training_progress.current_epoch = progress_data['epoch']
                    self.training_progress.current_loss = progress_data['train_loss']
                    self.training_progress.current_accuracy = progress_data['train_acc']
                    self.training_progress.message = f"Training epoch {progress_data['epoch']}/{progress_data['total_epochs']}"
            
            # Train the model
            training_results = trainer.train_model(
                architecture=architecture,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                progress_callback=progress_callback
            )
            
            # Training completed
            if self.training_progress.status == TrainingStatus.TRAINING:
                self.training_progress.status = TrainingStatus.COMPLETED
                self.training_progress.current_accuracy = training_results['best_val_acc']
                self.training_progress.validation_accuracy = training_results['final_val_acc']
                self.training_progress.message = "Training completed successfully"
                
                # Add to metadata
                model_name = training_results['model_name']
                logger.info(f"Adding model {model_name} to metadata")
                
                self.metadata["models"][model_name] = {
                    "version": "1.0",
                    "architecture": architecture.value,
                    "created_at": datetime.now().isoformat(),
                    "accuracy": training_results['best_val_acc'],
                    "loss": training_results['training_history']['val_loss'][-1] if training_results['training_history']['val_loss'] else None,
                    "training_config": {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size
                    },
                    "class_to_idx": training_results['class_to_idx'],
                    "num_classes": training_results['num_classes']
                }
                
                logger.info(f"Model added to metadata. Total models: {len(self.metadata['models'])}")
                logger.info(f"Models in metadata: {list(self.metadata['models'].keys())}")
                
                self._save_metadata()
                logger.info(f"Training completed. Model saved: {model_name}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.training_progress.status = TrainingStatus.FAILED
            self.training_progress.message = str(e)
    
    def get_training_status(self) -> TrainingProgress:
        """Get current training status.
        
        Returns:
            Current training progress
        """
        return self.training_progress
    
    def classify_image(self, image_path: str) -> Optional[ClassificationResult]:
        """Classify an image using the active model.
        
        Args:
            image_path: Path to the image to classify
            
        Returns:
            Classification result or None if failed
        """
        if not self.active_model:
            logger.error("No active model for classification")
            return None
        
        try:
            # Load the active model
            model_path = self.models_path / f"{self.active_model}.pth"
            if not model_path.exists():
                logger.error(f"Model file {model_path} not found")
                return None
            
            # Load model and metadata
            model, checkpoint = load_trained_model(str(model_path))
            class_to_idx = checkpoint.get('class_to_idx', {})
            
            # Classify the image
            predicted_class, confidence = classify_image_with_model(
                model=model,
                image_path=image_path,
                class_to_idx=class_to_idx,
                device='cpu'  # Use CPU for inference
            )
            
            result = ClassificationResult(
                class_name=predicted_class,
                confidence=confidence,
                image_path=image_path,
                active_model=self.active_model
            )
            
            logger.info(f"Classified image {image_path} as {predicted_class} with confidence {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return None
    
    def classify_image_from_memory(self, image_data: bytes) -> Optional[ClassificationResult]:
        """Classify an image from memory using the active model.
        
        Args:
            image_data: Raw bytes of the image
            
        Returns:
            Classification result or None if failed
        """
        if not self.active_model:
            logger.error("No active model for classification")
            return None
        
        try:
            # Load the active model
            model_path = self.models_path / f"{self.active_model}.pth"
            if not model_path.exists():
                logger.error(f"Model file {model_path} not found")
                return None
            
            # Load model and metadata
            model, checkpoint = load_trained_model(str(model_path))
            class_to_idx = checkpoint.get('class_to_idx', {})
            
            # Classify the image
            predicted_class, confidence = classify_image_from_memory(
                model=model,
                image=image_data,
                class_to_idx=class_to_idx,
                device='cpu'  # Use CPU for inference
            )
            
            result = ClassificationResult(
                class_name=predicted_class,
                confidence=confidence,
                image_path="memory_image", # Indicate it's from memory
                active_model=self.active_model
            )
            
            logger.info(f"Classified image from memory as {predicted_class} with confidence {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying image from memory: {e}")
            return None
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model for classification.
        
        Args:
            model_name: Name of the model to activate
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.metadata["models"]:
            logger.error(f"Model {model_name} not found")
            return False
        
        model_path = self.models_path / f"{model_name}.pth"
        if not model_path.exists():
            logger.error(f"Model file {model_path} not found")
            return False
        
        self.active_model = model_name
        self._save_metadata()
        logger.info(f"Set active model: {model_name}")
        return True
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.metadata["models"]:
            logger.error(f"Model {model_name} not found in metadata")
            return False
        
        try:
            # Delete model file
            model_path = self.models_path / f"{model_name}.pth"
            logger.info(f"Attempting to delete model file: {model_path}")
            
            if model_path.exists():
                logger.info(f"Model file exists, size: {model_path.stat().st_size} bytes")
                try:
                    import os
                    os.remove(str(model_path))
                    logger.info(f"Model file deleted successfully using os.remove")
                except Exception as remove_error:
                    logger.error(f"Failed to delete with os.remove: {remove_error}")
                    try:
                        model_path.unlink()
                        logger.info(f"Model file deleted successfully using unlink")
                    except Exception as unlink_error:
                        logger.error(f"Failed to delete with unlink: {unlink_error}")
                        raise Exception(f"Could not delete file: {remove_error}, {unlink_error}")
            else:
                logger.warning(f"Model file does not exist: {model_path}")
            
            # Remove from metadata
            logger.info(f"Removing model {model_name} from metadata")
            del self.metadata["models"][model_name]
            
            # If this was the active model, clear it
            if self.active_model == model_name:
                logger.info(f"Model {model_name} was active, clearing active model")
                self.active_model = None
            
            self._save_metadata()
            logger.info(f"Deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            return False
    
    # Production Mode Methods
    
    def start_production_mode(self) -> bool:
        """Start production mode for continuous classification.
        
        Returns:
            True if production mode started successfully, False otherwise
        """
        if not self.active_model:
            logger.error("No active model for production mode")
            return False
        
        if self.production_mode:
            logger.warning("Production mode already running")
            return True
        
        try:
            self.production_mode = True
            self.production_stats.start_time = datetime.now()
            self.production_stats.last_processed = datetime.now()
            
            # Start production processing thread
            self.production_thread = threading.Thread(target=self._production_worker)
            self.production_thread.daemon = True
            self.production_thread.start()
            
            logger.info("Production mode started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting production mode: {e}")
            self.production_mode = False
            return False
    
    def stop_production_mode(self) -> bool:
        """Stop production mode.
        
        Returns:
            True if production mode stopped successfully, False otherwise
        """
        if not self.production_mode:
            logger.warning("Production mode not running")
            return True
        
        try:
            self.production_mode = False
            
            # Wait for production thread to finish
            if self.production_thread and self.production_thread.is_alive():
                self.production_thread.join(timeout=5.0)
            
            # Save final statistics
            self._save_production_stats()
            
            logger.info("Production mode stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping production mode: {e}")
            return False
    
    def add_to_production_queue(self, image_path: str) -> bool:
        """Add an image to the production processing queue.
        
        Args:
            image_path: Path to the image to classify
            
        Returns:
            True if added successfully, False otherwise
        """
        if not self.production_mode:
            logger.error("Production mode not running")
            return False
        
        try:
            self.production_queue.put(image_path)
            logger.info(f"Added image to production queue: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Error adding image to production queue: {e}")
            return False
    
    def _production_worker(self) -> None:
        """Background worker for production mode processing."""
        logger.info("Production worker started")
        
        while self.production_mode:
            try:
                # Get image from queue with timeout
                try:
                    image_path = self.production_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the image
                start_time = time.time()
                result = self.classify_image(image_path)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.production_stats.total_processed += 1
                self.production_stats.last_processed = datetime.now()
                
                if result:
                    self.production_stats.successful_classifications += 1
                    
                    # Update confidence distribution
                    if result.confidence >= 0.9:
                        self.production_stats.confidence_distribution["high"] += 1
                    elif result.confidence >= 0.7:
                        self.production_stats.confidence_distribution["medium"] += 1
                    else:
                        self.production_stats.confidence_distribution["low"] += 1
                    
                    # Check if confidence meets threshold
                    if result.confidence < self.production_settings.confidence_threshold:
                        logger.warning(f"Low confidence classification: {result.class_name} ({result.confidence:.3f})")
                    
                    # Save result if enabled
                    if self.production_settings.save_results:
                        self._save_production_result(result, processing_time)
                    
                else:
                    self.production_stats.failed_classifications += 1
                    logger.error(f"Failed to classify image: {image_path}")
                
                # Update average processing time
                if self.production_stats.total_processed > 1:
                    self.production_stats.average_processing_time = (
                        (self.production_stats.average_processing_time * (self.production_stats.total_processed - 1) + processing_time) /
                        self.production_stats.total_processed
                    )
                else:
                    self.production_stats.average_processing_time = processing_time
                
                # Check processing time limit
                if processing_time > self.production_settings.max_processing_time:
                    logger.warning(f"Slow processing time: {processing_time:.2f}s for {image_path}")
                
                # Save statistics periodically
                if self.production_stats.total_processed % 10 == 0:
                    self._save_production_stats()
                
            except Exception as e:
                logger.error(f"Error in production worker: {e}")
                self.production_stats.failed_classifications += 1
        
        logger.info("Production worker stopped")
    
    def _save_production_result(self, result: ClassificationResult, processing_time: float) -> None:
        """Save production classification result.
        
        Args:
            result: Classification result
            processing_time: Processing time in seconds
        """
        try:
            result_data = {
                "timestamp": result.timestamp.isoformat(),
                "class_name": result.class_name,
                "confidence": result.confidence,
                "image_path": result.image_path,
                "active_model": result.active_model,
                "processing_time": processing_time
            }
            
            # Create filename based on timestamp
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            result_file = self.production_results_path / f"result_{timestamp_str}.json"
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Saved production result: {result_file}")
            
            # Also update statistics if not in production mode
            if not self.production_mode:
                self.production_stats.total_processed += 1
                self.production_stats.successful_classifications += 1
                self.production_stats.last_processed = datetime.now()
                
                # Update confidence distribution
                if result.confidence >= 0.9:
                    self.production_stats.confidence_distribution["high"] += 1
                elif result.confidence >= 0.7:
                    self.production_stats.confidence_distribution["medium"] += 1
                else:
                    self.production_stats.confidence_distribution["low"] += 1
                
                # Save statistics
                self._save_production_stats()
            
        except Exception as e:
            logger.error(f"Error saving production result: {e}")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get production mode status and statistics.
        
        Returns:
            Dictionary with production status and statistics
        """
        status = {
            "production_mode": self.production_mode,
            "active_model": self.active_model,
            "settings": asdict(self.production_settings),
            "stats": asdict(self.production_stats),
            "queue_size": self.production_queue.qsize()
        }
        logger.info(f"Production status: {status}")
        return status
    
    def update_production_settings(self, settings: Dict[str, Any]) -> bool:
        """Update production mode settings.
        
        Args:
            settings: Dictionary with new settings
            
        Returns:
            True if settings updated successfully, False otherwise
        """
        try:
            # Update settings
            for key, value in settings.items():
                if hasattr(self.production_settings, key):
                    setattr(self.production_settings, key, value)
            
            logger.info("Production settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Error updating production settings: {e}")
            return False
    
    def reset_production_stats(self) -> bool:
        """Reset production statistics.
        
        Returns:
            True if statistics reset successfully, False otherwise
        """
        try:
            self.production_stats = ProductionStats()
            self._save_production_stats()
            
            logger.info("Production statistics reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting production statistics: {e}")
            return False
    
    def get_production_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent production results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent production results
        """
        try:
            results = []
            result_files = sorted(
                self.production_results_path.glob("result_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for result_file in result_files[:limit]:
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        results.append(result_data)
                except Exception as e:
                    logger.error(f"Error reading result file {result_file}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting production results: {e}")
            return [] 