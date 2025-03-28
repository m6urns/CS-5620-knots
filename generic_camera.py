import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class Frame:
    """Generic frame data"""
    data: np.ndarray
    timestamp: int
    frame_number: int
    
    @property
    def height(self):
        return self.data.shape[0]
        
    @property
    def width(self):
        return self.data.shape[1]

class GenericCamera:
    """Camera interface using OpenCV"""
    
    @staticmethod
    def list_available_cameras(max_cameras=10):
        """List all available OpenCV cameras on the system
        
        Args:
            max_cameras: Maximum number of camera indices to check
            
        Returns:
            List of dictionaries with camera information
        """
        available_cameras = []
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        available_cameras.append({
                            'index': i,
                            'working': True,
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': fps
                        })
                    else:
                        available_cameras.append({
                            'index': i,
                            'working': False
                        })
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
                continue
        return available_cameras

    def __init__(self, camera_index: int = 0, resolution: tuple = (640, 480), fps: int = 30):
        """Initialize camera interface
        
        Args:
            camera_index: Index of the camera to use
            resolution: Desired resolution as (width, height)
            fps: Desired frames per second
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.frame_count = 0
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the camera
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            print(f"Opening camera with index {self.camera_index}...")
            
            # Try to open with default backend
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                # Try with specific backend
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Set camera properties
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Give the camera time to initialize
            time.sleep(1.0)
            
            # Warm up the camera by reading a few frames
            for _ in range(5):
                self.cap.read()
                time.sleep(0.1)
                
            # Get and print camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def get_frame(self) -> Optional[Frame]:
        """Get the next frame from the camera
        
        Returns:
            Frame object or None if no frame is available
        """
        if not self.is_initialized or not self.cap or not self.cap.isOpened():
            return None
        
        # Try to read frame with retry
        for attempt in range(3):  # Try up to 3 times
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                timestamp = int(time.time() * 1_000_000)  # Use system time in microseconds
                return Frame(frame, timestamp, self.frame_count)
                
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(0.01)
        
        return None
    
    def close(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
        print("Camera resources released")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()