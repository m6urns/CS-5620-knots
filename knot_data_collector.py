import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from imi_wrapper import ImiCamera, StreamType
from imi_visualization import FrameVisualizer, VisualizationConfig, ColorMap

@dataclass
class KnotSample:
    """Metadata for an overhand knot sample"""
    stage: str  # loose, loop, complete, tightened
    capture_timestamp: str
    notes: Optional[str] = None
    rgb_only: bool = False  # Flag to mark RGB-only samples

class OverhandKnotCollector:
    """Tool for collecting overhand knot RGB-D data"""
    
    STAGES = ["loose", "loop", "complete", "tightened"]
    
    def __init__(self, color_index: int = 4, base_path: str = "overhand_knot_dataset", rgb_only: bool = False):
        """Initialize data collector
        
        Args:
            color_index: Index of the color camera to use
            base_path: Base directory for dataset storage
            rgb_only: Use RGB-only mode (no depth data)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.color_index = color_index
        self.rgb_only = rgb_only
        
        # Setup visualization with info overlay for display only
        self.viz_config = VisualizationConfig(
            min_depth=100,
            max_depth=1000,
            auto_range=True,
            colormap=ColorMap.TURBO,
            show_histogram=True,
            show_info=False,  # Don't show info overlay in saved images
            view_mode="side-by-side" if not rgb_only else "color-only",
            window_width=800,
            window_height=600
        )
        self.viz = FrameVisualizer(self.viz_config)
        
        # Initialize camera
        self.camera = None
        self.current_stage = "loose"
        
        # Initialize or load sample counts
        self.counts_path = self.base_path / "sample_counts.json"
        self.sample_counts = self._load_counts()
        
    def _load_counts(self) -> Dict[str, int]:
        """Load or initialize sample counts for each stage"""
        if self.counts_path.exists():
            with open(self.counts_path, 'r') as f:
                return json.load(f)
        return {stage: 0 for stage in self.STAGES}
        
    def _save_counts(self):
        """Save sample counts"""
        with open(self.counts_path, 'w') as f:
            json.dump(self.sample_counts, f, indent=2)
            
    def _save_sample(self, rgb_frame, depth_frame=None, sample: KnotSample = None):
        """Save a synchronized RGB-D sample with metadata
        
        In RGB-only mode, depth_frame may be None
        """
        # Create sample directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = self.base_path / sample.stage / timestamp
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw RGB frame without overlays
        cv2.imwrite(str(sample_dir / "rgb.png"), rgb_frame.data)
        
        # Update sample metadata with RGB-only flag
        sample.rgb_only = self.rgb_only
        
        # Save depth data if available (not in RGB-only mode)
        if not self.rgb_only and depth_frame is not None:
            np.save(str(sample_dir / "depth.npy"), depth_frame.data)
            
            # Create clean depth visualization without overlays
            depth_colormap, _ = self.viz.visualize_depth(depth_frame.data)
            if depth_colormap is not None:
                # Save clean depth visualization
                cv2.imwrite(str(sample_dir / "depth_viz.png"), depth_colormap)
                
                # Resize images to match before combining
                rgb_height, rgb_width = rgb_frame.data.shape[:2]
                depth_height, depth_width = depth_colormap.shape[:2]
                
                # Create clean combined visualization
                depth_viz_resized = cv2.resize(depth_colormap, 
                                             (int(depth_width * rgb_height / depth_height), rgb_height))
                combined = np.hstack((rgb_frame.data, depth_viz_resized))
                cv2.imwrite(str(sample_dir / "combined.png"), combined)
        
        # Save metadata
        metadata_dict = {
            "stage": sample.stage,
            "capture_timestamp": sample.capture_timestamp,
            "notes": sample.notes,
            "rgb_only": sample.rgb_only,
            "rgb_frame_number": rgb_frame.frame_number,
        }
        
        # Add depth info if available
        if not self.rgb_only and depth_frame is not None:
            metadata_dict.update({
                "depth_frame_number": depth_frame.frame_number,
                "rgb_timestamp": rgb_frame.timestamp,
                "depth_timestamp": depth_frame.timestamp
            })
        
        with open(sample_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        # Update counts
        self.sample_counts[sample.stage] += 1
        self._save_counts()
        
        return sample_dir
        
    def start_collection(self):
        """Start interactive data collection session"""
        try:
            print("\nInitializing camera...")
            self.camera = ImiCamera(color_index=self.color_index)
            self.camera.initialize()
            
            # Open color stream
            self.camera.open_stream(StreamType.COLOR)
            print(f"Color stream opened successfully (using index {self.color_index})")
            
            # Open depth stream if not in RGB-only mode
            if not self.rgb_only:
                self.camera.open_stream(StreamType.DEPTH)
                print("Depth stream opened successfully")
            else:
                print("Running in RGB-only mode (no depth data)")
            
            print("\nOverhand Knot Data Collection")
            print("============================")
            print("\nCurrent sample counts:")
            for stage in self.STAGES:
                print(f"  {stage}: {self.sample_counts[stage]}")
            
            print("\nControls:")
            print("  'space': Capture sample")
            print("  's': Cycle knot stage")
            print("  'n': Add note to next capture")
            if not self.rgb_only:
                print("  'v': Toggle view mode")
                print("  'r': Toggle auto-range")
            print("  'q': Quit")
            
            note_for_next = None
            
            while True:
                # Get color frame
                color_frame = self.camera.get_frame(StreamType.COLOR)
                
                # Get depth frame if not in RGB-only mode
                depth_frame = None
                if not self.rgb_only:
                    depth_frame = self.camera.get_frame(StreamType.DEPTH)
                
                if color_frame is not None and (self.rgb_only or depth_frame is not None):
                    # Show frames and current stage
                    color_viz = color_frame.data.copy()
                    
                    # Add stage overlay
                    cv2.putText(color_viz, f"Stage: {self.current_stage}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0), 2)
                    cv2.putText(color_viz, f"Samples: {self.sample_counts[self.current_stage]}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0), 2)
                    
                    # Add mode indicator
                    if self.rgb_only:
                        cv2.putText(color_viz, "Mode: RGB-only", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 165, 255), 2)
                              
                    # Add controls overlay
                    controls = [
                        "Controls:",
                        "SPACE: Capture sample",
                        "S: Cycle stage",
                        "N: Add note",
                    ]
                    
                    if not self.rgb_only:
                        controls.extend([
                            "V: Toggle view",
                            "R: Auto-range",
                        ])
                        
                    controls.append("Q: Quit")
                    
                    y_pos = 120
                    for control in controls:
                        cv2.putText(color_viz, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (255, 255, 255), 2)  # White with black outline
                        cv2.putText(color_viz, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 0), 1)
                        y_pos += 25
                    
                    # Show frames
                    if self.rgb_only:
                        # In RGB-only mode, we only show the color frame
                        key = self.viz.show_rgb_only(color_viz)
                    else:
                        # In dual-stream mode, we show both color and depth
                        depth_viz = depth_frame.data.copy()
                        key = self.viz.show(depth_viz, color_viz)
                    
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to capture
                        sample = KnotSample(
                            stage=self.current_stage,
                            capture_timestamp=datetime.now().isoformat(),
                            notes=note_for_next,
                            rgb_only=self.rgb_only
                        )
                        
                        sample_dir = self._save_sample(color_frame, depth_frame, sample)
                        print(f"\nSaved {self.current_stage} sample to {sample_dir}")
                        print(f"Total {self.current_stage} samples: {self.sample_counts[self.current_stage]}")
                        note_for_next = None
                        
                    elif key == ord('s'):
                        # Cycle stage
                        current_idx = self.STAGES.index(self.current_stage)
                        self.current_stage = self.STAGES[(current_idx + 1) % len(self.STAGES)]
                        print(f"\nStage: {self.current_stage}")
                        print(f"Samples: {self.sample_counts[self.current_stage]}")
                        
                    elif key == ord('n'):
                        note = input("\nEnter note for next capture: ")
                        note_for_next = note if note.strip() else None
                        
                    elif not self.rgb_only and key == ord('v'):
                        # Cycle view mode (not available in RGB-only mode)
                        modes = ["side-by-side", "overlay"]
                        current_idx = modes.index(self.viz_config.view_mode) if self.viz_config.view_mode in modes else 0
                        self.viz_config.view_mode = modes[(current_idx + 1) % len(modes)]
                        print(f"\nView mode: {self.viz_config.view_mode}")
                        
                    elif not self.rgb_only and key == ord('r'):
                        # Toggle auto-range (not available in RGB-only mode)
                        self.viz_config.auto_range = not self.viz_config.auto_range
                        print(f"\nAuto-range: {self.viz_config.auto_range}")
                        
        finally:
            if self.camera:
                self.camera.close()
            if self.viz:
                self.viz.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect overhand knot RGB-D data')
    parser.add_argument('--color-index', type=int, default=4,
                       help='Index of the color camera to use (default: 4)')
    parser.add_argument('--rgb-only', action='store_true',
                       help='Use RGB-only mode (no depth data)')
    args = parser.parse_args()
    
    collector = OverhandKnotCollector(color_index=args.color_index, rgb_only=args.rgb_only)
    collector.start_collection()

if __name__ == "__main__":
    main()