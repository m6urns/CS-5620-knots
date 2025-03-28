import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

class ColorMap(Enum):
    """Available colormaps for depth visualization"""
    JET = cv2.COLORMAP_JET
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    MAGMA = cv2.COLORMAP_MAGMA
    PLASMA = cv2.COLORMAP_PLASMA
    TURBO = cv2.COLORMAP_TURBO

@dataclass
class VisualizationConfig:
    """Configuration for depth and color visualization"""
    min_depth: float = 0.0
    max_depth: float = 10000.0
    auto_range: bool = True
    colormap: ColorMap = ColorMap.TURBO
    show_histogram: bool = True
    show_info: bool = True
    window_name: str = "Depth"
    color_window: str = "Color"
    rgb_only_window: str = "RGB Camera"  # New window name for RGB-only mode
    histogram_window: str = "Histogram"
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    view_mode: str = "side-by-side"  # or "overlay" or "color-only"
    window_width: int = 640
    window_height: int = 480
    vertical_shift: int = 71    
    horizontal_shift: int = 45   
    alignment_mode: bool = False
    rgb_only: bool = False  # New flag for RGB-only mode

class FrameVisualizer:
    """Visualizer for depth and RGB frames"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with optional config"""
        self.config = config or VisualizationConfig()
        self.last_frame_time = time.time()
        self.running_fps = 0
        self.windows_created = False
        self._create_windows()
        
    def _create_windows(self):
        """Create and position visualization windows"""
        if not self.windows_created:
            screen_width = 1920  # Assume standard screen width
            
            # For RGB-only mode, create just one window
            if self.config.rgb_only or self.config.view_mode == "color-only":
                cv2.namedWindow(self.config.rgb_only_window, 
                              cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow(self.config.rgb_only_window, 
                               self.config.window_width, self.config.window_height)
                cv2.moveWindow(self.config.rgb_only_window, 
                             screen_width//2 - self.config.window_width//2, 0)
            else:
                # Create depth window
                cv2.namedWindow(self.config.window_name, 
                              cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow(self.config.window_name, 
                               self.config.window_width, self.config.window_height)
                
                # Create color window if in side-by-side mode
                if self.config.view_mode == "side-by-side":
                    cv2.namedWindow(self.config.color_window, 
                                  cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(self.config.color_window,
                                   self.config.window_width, self.config.window_height)
                
                if self.config.show_histogram:
                    cv2.namedWindow(self.config.histogram_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.config.histogram_window, 400, 200)
                
                # Position windows
                if self.config.view_mode == "side-by-side":
                    cv2.moveWindow(self.config.window_name, 0, 0)
                    cv2.moveWindow(self.config.color_window, 
                                 self.config.window_width + 30, 0)
                    if self.config.show_histogram:
                        cv2.moveWindow(self.config.histogram_window, 
                                     2 * self.config.window_width + 60, 0)
                else:  # overlay mode
                    cv2.moveWindow(self.config.window_name, 
                                 screen_width//2 - self.config.window_width//2, 0)
            
            self.windows_created = True

    def _create_info_overlay(self, depth_frame: Optional[np.ndarray] = None, 
                             valid_mask: Optional[np.ndarray] = None,
                             color_frame: Optional[np.ndarray] = None) -> List[str]:
        """Create information overlay with additional color stats"""
        info = [f"FPS: {self.running_fps:.1f}"]
        
        # Add RGB-only mode indicator
        if self.config.rgb_only:
            info.append("Mode: RGB-only")
        else:
            # Add depth-related info
            if depth_frame is not None and valid_mask is not None:
                info.extend([
                    f"Depth range: {self.config.min_depth:.0f}-{self.config.max_depth:.0f}mm",
                    f"Valid pixels: {(np.sum(valid_mask)/valid_mask.size*100):.1f}%",
                    f"Mean depth: {np.mean(depth_frame[valid_mask]):.0f}mm",
                    f"View mode: {self.config.view_mode}",
                    "Auto range: ON" if self.config.auto_range else "Auto range: OFF"
                ])
        
        if color_frame is not None:
            info.extend([
                f"Color resolution: {color_frame.shape[1]}x{color_frame.shape[0]}",
                f"Mean brightness: {np.mean(color_frame):.1f}"
            ])
        
        # Add controls based on mode
        if self.config.rgb_only:
            info.extend([
                "'s': Save frame",
                "'q': Quit"
            ])
        else:
            info.extend([
                "'v': Toggle view mode",
                "'r': Toggle auto range",
                "'h': Toggle histogram",
                "'c': Change colormap",
                "'s': Save frame",
                "'q': Quit"
            ])
            
        return info
    
    def show_rgb_only(self, rgb_frame: np.ndarray):
        """Display RGB frame only without depth data
        
        Args:
            rgb_frame: RGB camera frame
            
        Returns:
            Key pressed (if any) during visualization
        """
        if rgb_frame is None:
            return None

        self._update_fps()
        
        # Add information overlay
        if self.config.show_info:
            rgb_viz = rgb_frame.copy()
            info_text = self._create_info_overlay(color_frame=rgb_frame)
            y = 30
            for text in info_text:
                cv2.putText(rgb_viz, text, (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(rgb_viz, text, (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y += 25
        else:
            rgb_viz = rgb_frame
        
        # Display the RGB frame
        cv2.imshow(self.config.rgb_only_window, rgb_viz)
        
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)
        return key
    
    def show(self, depth_frame: np.ndarray, rgb_frame: Optional[np.ndarray] = None):
        """Display depth and RGB frames with 2D alignment correction
        
        Falls back to RGB-only display if in RGB-only mode
        """
        # If in RGB-only mode or depth frame is None, use RGB-only display
        if self.config.rgb_only or depth_frame is None:
            if rgb_frame is not None:
                return self.show_rgb_only(rgb_frame)
            return cv2.waitKey(1) & 0xFF
        
        depth_colormap, normalized = self.visualize_depth(depth_frame)
        
        if depth_colormap is not None:
            if rgb_frame is not None:
                # Resize color frame to match depth
                rgb_resized = cv2.resize(rgb_frame, 
                                    (depth_colormap.shape[1], depth_colormap.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                
                # Always apply alignment correction
                rows, cols = depth_colormap.shape[:2]
                shift_matrix = np.float32([[1, 0, self.config.horizontal_shift], 
                                        [0, 1, self.config.vertical_shift]])
                
                depth_colormap = cv2.warpAffine(depth_colormap, 
                                            shift_matrix, 
                                            (cols, rows),
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=[0, 0, 0])
                
                if self.config.view_mode == "overlay":
                    alpha = 0.7
                    overlay = cv2.addWeighted(depth_colormap, alpha, 
                                            rgb_resized, 1-alpha, 0)
                    cv2.imshow(self.config.window_name, overlay)
                else:
                    cv2.imshow(self.config.window_name, depth_colormap)
                    cv2.imshow(self.config.color_window, rgb_resized)
            else:
                cv2.imshow(self.config.window_name, depth_colormap)
            
            if self.config.show_histogram:
                hist_img = self._create_histogram(depth_frame)
                cv2.imshow(self.config.histogram_window, hist_img)
                
            # Show alignment info window only when alignment adjustment mode is enabled
            if self.config.alignment_mode:
                info_img = np.zeros((100, 400, 3), dtype=np.uint8)
                cv2.putText(info_img, f"Alignment Adjustment Mode: ENABLED", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(info_img, f"Vertical shift: {self.config.vertical_shift}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(info_img, f"Horizontal shift: {self.config.horizontal_shift}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(info_img, "WASD keys to adjust alignment",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Alignment Info", info_img)
            elif cv2.getWindowProperty("Alignment Info", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Alignment Info")
                
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)
        return key

    def _handle_key(self, key: int):
        """Handle keyboard input including alignment adjustments"""
        if key == ord('r') and not self.config.rgb_only:  
            self.config.auto_range = not self.config.auto_range
        elif key == ord('h') and not self.config.rgb_only:
            self.config.show_histogram = not self.config.show_histogram
            if not self.config.show_histogram:
                cv2.destroyWindow(self.config.histogram_window)
        elif key == ord('c') and not self.config.rgb_only:
            colormaps = list(ColorMap)
            current_idx = colormaps.index(self.config.colormap)
            self.config.colormap = colormaps[(current_idx + 1) % len(colormaps)]
        elif key == ord('v') and not self.config.rgb_only:
            self.config.view_mode = "overlay" if self.config.view_mode == "side-by-side" else "side-by-side"
            cv2.destroyAllWindows()
            self.windows_created = False
            self._create_windows()
        elif key == ord('m') and not self.config.rgb_only:  # New hotkey to toggle alignment mode
            self.config.alignment_mode = not self.config.alignment_mode
            print(f"Alignment mode {'enabled' if self.config.alignment_mode else 'disabled'}")
            if not self.config.alignment_mode and cv2.getWindowProperty("Alignment Info", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Alignment Info")
        
        # Only process WASD keys if alignment mode is enabled and not in RGB-only mode
        if self.config.alignment_mode and not self.config.rgb_only:
            if key == ord('w'):
                self.config.vertical_shift -= 2
                print(f"Vertical shift: {self.config.vertical_shift}")
            elif key == ord('s'):
                self.config.vertical_shift += 2
                print(f"Vertical shift: {self.config.vertical_shift}")
            elif key == ord('d'):
                self.config.horizontal_shift += 2
                print(f"Horizontal shift: {self.config.horizontal_shift}")
            elif key == ord('a'):
                self.config.horizontal_shift -= 2
                print(f"Horizontal shift: {self.config.horizontal_shift}")
            
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.running_fps = 0.9 * self.running_fps + 0.1 * (1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        
    def _create_histogram(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create depth histogram visualization"""
        hist = cv2.calcHist([depth_frame], [0], None, [256], [0, self.config.max_depth])
        hist_img = np.zeros((200, 256), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
        
        # Draw histogram bars
        for i in range(256):
            if hist[i] > 0:
                cv2.line(hist_img, (i, 200), (i, 200 - int(hist[i])), 255)
        
        # Add color
        return cv2.applyColorMap(hist_img, cv2.COLORMAP_HOT)
        
    def visualize_depth(self, depth_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create visualization of depth frame"""
        if depth_frame is None:
            return None, None

        self._update_fps()
        
        # Create mask for valid depth values
        valid_mask = (depth_frame > 0)
        if self.config.auto_range and np.sum(valid_mask) > 0:
            self.config.min_depth = np.percentile(depth_frame[valid_mask], 
                                                self.config.percentile_min)
            self.config.max_depth = np.percentile(depth_frame[valid_mask], 
                                                self.config.percentile_max)
        
        valid_mask &= (depth_frame >= self.config.min_depth) & (depth_frame <= self.config.max_depth)
        
        if np.sum(valid_mask) > 0:
            # Normalize valid depths
            normalized = np.zeros_like(depth_frame, dtype=np.float32)
            normalized[valid_mask] = ((depth_frame[valid_mask] - self.config.min_depth) * 255 / 
                                    (self.config.max_depth - self.config.min_depth))
            
            # Clip to 0-255 range and convert to uint8
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # Apply colormap
            depth_colormap = cv2.applyColorMap(normalized, self.config.colormap.value)
            
            # Make invalid pixels black
            depth_colormap[~valid_mask] = 0
            
            # Scale image to fit window (maintain aspect ratio)
            window_width = 1280
            window_height = 720
            scale = min(window_width / depth_colormap.shape[1],
                       window_height / depth_colormap.shape[0])
            dim = (int(depth_colormap.shape[1] * scale),
                  int(depth_colormap.shape[0] * scale))
            depth_colormap = cv2.resize(depth_colormap, dim, 
                                      interpolation=cv2.INTER_AREA)
            
            # Add information overlay
            if self.config.show_info:
                info_text = self._create_info_overlay(depth_frame, valid_mask)
                y = 30
                for text in info_text:
                    cv2.putText(depth_colormap, text, (10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, text, (10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    y += 25
            
            return depth_colormap, normalized
        return None, None
    
    def save_frame(self, rgb_frame: Optional[np.ndarray] = None, 
                   depth_frame: Optional[np.ndarray] = None, 
                   depth_colormap: Optional[np.ndarray] = None):
        """Save current frames to disk
        
        Updated to handle RGB-only mode
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save depth data if available
        if depth_frame is not None:
            np.save(f'depth_frame_{timestamp}.npy', depth_frame)
        
        if depth_colormap is not None:
            cv2.imwrite(f'depth_viz_{timestamp}.png', depth_colormap)
            
        # Save RGB data if available
        if rgb_frame is not None:
            cv2.imwrite(f'rgb_frame_{timestamp}.png', rgb_frame)
            
        print(f"Frames saved with timestamp {timestamp}")
        
    def close(self):
        """Clean up visualization windows"""
        cv2.destroyAllWindows()
        self.windows_created = False