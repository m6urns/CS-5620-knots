import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
from torchvision import transforms
from datetime import datetime
import threading
import queue
import time
import argparse
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from pathlib import Path

from knots.knot_classifier import KnotClassifier
from knots.knot_definition import KnotDefinition
from generic_camera import GenericCamera
from visualization import VisualizationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """API settings model with explicit parameter names and descriptions"""
    # Camera settings
    camera_index: int = Field(
        default=0,
        description="Index of the camera to use",
        ge=0
    )
    width: int = Field(
        default=640,
        description="Camera resolution width",
        ge=320
    )
    height: int = Field(
        default=480,
        description="Camera resolution height",
        ge=240
    )
    fps: int = Field(
        default=30,
        description="Camera frames per second",
        ge=10
    )
    
    # Model settings
    model_path: str = Field(
        default='best_model.pth',
        description="Path to the trained model weights file"
    )
    knot_def_path: Optional[str] = Field(
        default=None,
        description="Path to the knot definition file (if not included in model)"
    )
    confidence_threshold: float = Field(
        default=0.4,
        description="Minimum confidence threshold for classifications",
        ge=0.0,
        le=1.0
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind the server"
    )
    port: int = Field(
        default=8000,
        description="Port number for the server",
        ge=0,
        le=65535
    )

class KnotClassifierAPI:
    """API wrapper for knot classifier"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize API wrapper with settings"""
        self.settings = settings or Settings()
        self.latest_frame = None
        self.latest_classification = {
            "stage": "unknown",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        self.fps = 0
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Initialize transforms
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize knot definition and classifier
        self.knot_def = None
        self.stages = []
        
        try:
            # Try to load model with knot definition
            weights_path = Path(self.settings.model_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found at {self.settings.model_path}")
            
            # First try loading as a model that includes knot definition
            try:
                logger.info("Trying to load model with embedded knot definition...")
                self.classifier = KnotClassifier.load_with_knot_def(
                    self.settings.model_path, self.device)
                self.knot_def = self.classifier.knot_def
                
                if self.knot_def:
                    logger.info(f"Loaded knot definition from model: {self.knot_def.name}")
                    logger.info(f"Stages: {self.knot_def.stage_ids}")
                    self.stages = self.knot_def.stage_ids
                else:
                    raise ValueError("Model loaded but no knot definition found")
                    
            except Exception as e:
                logger.warning(f"Could not load model with knot definition: {str(e)}")
                
                # If separate knot definition provided, try loading it
                if self.settings.knot_def_path:
                    try:
                        knot_def_path = Path(self.settings.knot_def_path)
                        if not knot_def_path.exists():
                            raise FileNotFoundError(f"Knot definition not found at {self.settings.knot_def_path}")
                            
                        logger.info(f"Loading knot definition from {self.settings.knot_def_path}")
                        self.knot_def = KnotDefinition.from_file(self.settings.knot_def_path)
                        self.stages = self.knot_def.stage_ids
                        logger.info(f"Loaded knot definition: {self.knot_def.name}")
                        logger.info(f"Stages: {self.stages}")
                    except Exception as e:
                        logger.error(f"Error loading knot definition: {str(e)}")
                        # Fall back to default stages
                        self.stages = ["loose", "loop", "complete", "tightened"]
                        logger.info(f"Using default stages: {self.stages}")
                else:
                    # No knot definition available, fall back to default stages
                    self.stages = ["loose", "loop", "complete", "tightened"]
                    logger.info(f"Using default stages: {self.stages}")
                
                # Initialize model with number of classes based on stages
                logger.info(f"Initializing model with {len(self.stages)} classes")
                self.classifier = KnotClassifier(num_classes=len(self.stages))
                
                # Load model weights
                logger.info("Loading model weights...")
                self.classifier.load_state_dict(
                    torch.load(self.settings.model_path, map_location=self.device))
                
            # Move model to device and set to eval mode
            self.classifier.to(self.device)
            self.classifier.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize camera
        try:
            self.camera = GenericCamera(
                camera_index=self.settings.camera_index,
                resolution=(self.settings.width, self.settings.height),
                fps=self.settings.fps
            )
            
            if not self.camera.initialize():
                raise RuntimeError(f"Failed to initialize camera with index {self.settings.camera_index}")
                
            logger.info(f"Camera initialized with index {self.settings.camera_index}")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            raise
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.is_running = True
        self.processing_thread.start()
        
    def preprocess_frame(self, frame_data):
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.rgb_transform(rgb).unsqueeze(0).to(self.device)
        return rgb_tensor
        
    def process_frames(self):
        """Main processing loop"""
        last_time = time.time()
        frame_count = 0
        logger.info("Starting frame processing loop")
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                # Process frame if available
                if frame is not None:
                    # Update FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        self.fps = 30 / (current_time - last_time)
                        last_time = current_time
                        logger.info(f"Processing frames at {self.fps:.1f} FPS")
                    
                    # Process frame for visualization
                    display_frame = frame.data.copy()
                    
                    # Get model prediction
                    with torch.no_grad():
                        rgb_tensor = self.preprocess_frame(frame.data)
                        self.last_rgb_tensor = rgb_tensor
                        
                        outputs = self.classifier(rgb_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        predicted_idx = predicted.item()
                        
                        if confidence >= self.settings.confidence_threshold:
                            # Check if we're within range of valid stages
                            if predicted_idx < len(self.stages):
                                predicted_stage = self.stages[predicted_idx]
                            else:
                                logger.warning(f"Predicted index {predicted_idx} out of range for stages")
                                predicted_stage = "unknown"
                        else:
                            predicted_stage = "unknown"
                    
                    # Get stage name if knot definition is available
                    stage_name = predicted_stage
                    if self.knot_def and predicted_stage != "unknown":
                        try:
                            stage_obj = self.knot_def.get_stage(predicted_stage)
                            stage_name = stage_obj.name
                        except ValueError:
                            pass
                    
                    self.latest_classification = {
                        "stage": predicted_stage,
                        "stage_name": stage_name if stage_name != predicted_stage else None,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add text overlays to display frame
                    confidence_color = (0, 255, 0) if confidence >= self.settings.confidence_threshold else (0, 165, 255)
                    display_name = stage_name if stage_name != predicted_stage else predicted_stage
                    cv2.putText(display_frame, f"Stage: {display_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              confidence_color, 2)
                    cv2.putText(display_frame, f"FPS: {self.fps:.1f}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (255, 255, 255), 2)
                    
                    # Update frame queue for streaming
                    try:
                        self.frame_queue.put_nowait(display_frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(display_frame)
                        except queue.Empty:
                            pass
                            
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)
    
    def encode_frame(self):
        """Generator for MJPEG streaming"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    ret, encoded_frame = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               encoded_frame.tobytes() + b'\r\n')
            except queue.Empty:
                continue
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "camera_connected": self.camera is not None and self.camera.is_initialized,
            "model_loaded": hasattr(self, 'classifier'),
            "knot_definition": self.knot_def.name if self.knot_def else None,
            "stages": self.stages,
            "fps": self.fps,
            "timestamp": datetime.now().isoformat(),
            "camera_index": self.settings.camera_index,
            "resolution": f"{self.settings.width}x{self.settings.height}"
        }
    
    def get_classification(self) -> Dict:
        """Get latest classification with detailed probabilities"""
        with torch.no_grad():
            if hasattr(self, 'last_rgb_tensor'):
                outputs = self.classifier(self.last_rgb_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                all_probs = {stage: float(prob) for stage, prob in zip(self.stages, probabilities)}
            else:
                all_probs = {stage: 0.0 for stage in self.stages}
        
        # Get stage descriptions if knot definition is available
        stage_info = {}
        if self.knot_def:
            for stage_id in self.stages:
                try:
                    stage = self.knot_def.get_stage(stage_id)
                    stage_info[stage_id] = {
                        "name": stage.name,
                        "description": stage.description
                    }
                except ValueError:
                    stage_info[stage_id] = {
                        "name": stage_id,
                        "description": ""
                    }
        
        result = {
            "stage": self.latest_classification["stage"],
            "confidence": self.latest_classification["confidence"],
            "timestamp": self.latest_classification["timestamp"],
            "probabilities": all_probs
        }
        
        # Add stage info if available
        if stage_info:
            result["stage_info"] = stage_info
            
        # Add stage name if available
        if "stage_name" in self.latest_classification and self.latest_classification["stage_name"]:
            result["stage_name"] = self.latest_classification["stage_name"]
            
        return result
    
    def get_stream(self):
        """Get MJPEG stream"""
        return StreamingResponse(
            self.encode_frame(),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    
    def get_settings(self) -> Settings:
        """Get current settings"""
        return self.settings
    
    def update_settings(self, new_settings: Settings):
        """Update settings
        
        Note: Camera changes require restart
        """
        # Check if camera settings changed
        camera_changed = (
            self.settings.camera_index != new_settings.camera_index or
            self.settings.width != new_settings.width or
            self.settings.height != new_settings.height or
            self.settings.fps != new_settings.fps
        )
        
        # Check if model or knot definition changed
        model_changed = (
            self.settings.model_path != new_settings.model_path or
            self.settings.knot_def_path != new_settings.knot_def_path
        )
        
        # Update settings
        self.settings = new_settings
        
        # Return appropriate status
        if camera_changed:
            return {"status": "restart_required", "message": "Camera settings change requires restart"}
        elif model_changed:
            return {"status": "restart_required", "message": "Model or knot definition change requires restart"}
        return {"status": "success"}
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.camera:
            self.camera.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Knot Classifier API Server')
    
    # Camera settings
    parser.add_argument('--camera-index', type=int, default=0,
                      help='Index of the camera to use (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                      help='Camera resolution width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                      help='Camera resolution height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                      help='Camera frames per second (default: 30)')
    
    # Model settings
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                      help='Path to the trained model weights file (default: best_model.pth)')
    parser.add_argument('--knot-def-path', type=str, default=None,
                      help='Path to the knot definition file (default: None)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                      help='Minimum confidence threshold for classifications (default: 0.7)')
    
    # Server settings
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host address to bind the server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port number for the server (default: 8000)')
    
    return parser.parse_args()

# Global instance for API
classifier_api = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_api
    args = parse_args()
    settings = Settings(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_path=args.model_path,
        knot_def_path=args.knot_def_path,
        confidence_threshold=args.confidence_threshold,
        host=args.host,
        port=args.port
    )
    classifier_api = KnotClassifierAPI(settings=settings)
    yield
    if classifier_api:
        classifier_api.cleanup()

app = FastAPI(
    title="Knot Classifier API",
    description="API for real-time knot tying classification using RGB camera",
    lifespan=lifespan
)

@app.get("/status")
async def get_status():
    """
    Get current system status.
    
    Returns:
        dict: Contains information about:
            - Camera connection status
            - Model loading status
            - Current FPS
            - Current timestamp
            - Camera index and resolution
            - Knot definition (if available)
            - Available stages
    """
    return classifier_api.get_status()

@app.get("/classification")
async def get_classification():
    """
    Get the latest knot classification result.
    
    Returns:
        dict: Classification details including:
            - Predicted stage (e.g., "loose", "loop", "complete")
            - Stage name (if knot definition available)
            - Confidence score (0.0 to 1.0)
            - Timestamp of classification
            - Individual probability scores for each possible stage
            - Stage information (names and descriptions) if knot definition available
    """
    return classifier_api.get_classification()

@app.get("/stream")
async def get_stream():
    """
    Get live video stream of the classification visualization.
    
    Returns:
        StreamingResponse: MJPEG stream containing:
            - Camera feed with overlay
            - Current classification results
            - FPS counter
            
    Raises:
        500: If stream cannot be initialized or encounters an error
    """
    try:
        return classifier_api.get_stream()
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Stream error occurred"}
        )

@app.get("/settings")
async def get_settings():
    """
    Get current system settings.
    
    Returns:
        Settings: Complete settings object containing:
            - Camera settings (index, resolution, fps)
            - Model settings (path, knot definition path, confidence threshold)
            - Server settings (host, port)
    """
    return classifier_api.get_settings()

@app.post("/settings")
async def update_settings(settings: Settings):
    """
    Update system settings.
    
    Args:
        settings: Complete Settings object with new values.
            All fields are optional and only provided values will be updated.
    
    Returns:
        dict: Status of update operation
    """
    result = classifier_api.update_settings(settings)
    return result

def main():
    """Main entry point with command line arguments"""
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()