import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class VisualizationConfig:
    """Configuration for RGB visualization"""
    show_info: bool = True
    window_name: str = "Camera"
    window_width: int = 640
    window_height: int = 480
    
class FrameVisualizer:
    """Visualizer for RGB frames"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with optional config"""
        self.config = config or VisualizationConfig()
        self.last_frame_time = time.time()
        self.running_fps = 0
        self.windows_created = False
        self._create_windows()
    
    def _create_windows(self):
        """Create and position visualization window"""
        if not self.windows_created:
            cv2.namedWindow(self.config.window_name, 
                          cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow(self.config.window_name, 
                           self.config.window_width, self.config.window_height)
            
            # Center window on screen (approximate)
            screen_width = 1920  # Assume standard screen width
            cv2.moveWindow(self.config.window_name, 
                         screen_width//2 - self.config.window_width//2, 0)
            
            self.windows_created = True
    
    def _create_info_overlay(self, frame: np.ndarray) -> List[str]:
        """Create information overlay text"""
        info = [f"FPS: {self.running_fps:.1f}"]
        
        if frame is not None:
            info.extend([
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                f"Mean brightness: {np.mean(frame):.1f}"
            ])
        
        info.extend([
            # "'s': Save frame",
            # "'q': Quit"
        ])
            
        return info
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.running_fps = 0.9 * self.running_fps + 0.1 * (1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
    
    def show(self, frame: np.ndarray) -> int:
        """Display RGB frame
        
        Args:
            frame: RGB frame to display
            
        Returns:
            int: Key pressed (if any) during visualization
        """
        if frame is None:
            return cv2.waitKey(1) & 0xFF
        
        self._update_fps()
        
        # Add information overlay
        if self.config.show_info:
            display_frame = frame.copy()
            info_text = self._create_info_overlay(frame)
            # y = 30
            y = 100
            for text in info_text:
                cv2.putText(display_frame, text, (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, text, (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y += 25
        else:
            display_frame = frame
        
        # Display the RGB frame
        cv2.imshow(self.config.window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key, frame)
        return key
    
    def _handle_key(self, key: int, frame: np.ndarray):
        """Handle keyboard input"""
        # if key == ord('s'):
            # self.save_frame(frame)
        pass
    
    def save_frame(self, frame: np.ndarray):
        """Save current frame to disk
        
        Args:
            frame: RGB frame to save
        """
        if frame is None:
            return
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'rgb_frame_{timestamp}.png'
        
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    
    def close(self):
        """Clean up visualization windows"""
        cv2.destroyAllWindows()
        self.windows_created = False