import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

from generic_camera import GenericCamera, Frame
from visualization import FrameVisualizer, VisualizationConfig

@dataclass
class KnotSample:
    """Metadata for an overhand knot sample"""
    stage: str  # loose, loop, complete, tightened
    capture_timestamp: str
    notes: Optional[str] = None

class OverhandKnotCollector:
    """Tool for collecting overhand knot RGB data"""
    
    STAGES = ["loose", "loop", "complete", "tightened"]
    
    def __init__(self, camera_index: int = 0, base_path: str = "overhand_knot_dataset", 
                 resolution: tuple = (640, 480), fps: int = 30):
        """Initialize data collector
        
        Args:
            camera_index: Index of the camera to use
            base_path: Base directory for dataset storage
            resolution: Camera resolution as (width, height)
            fps: Camera frames per second
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        
        # Setup visualization with info overlay
        self.viz_config = VisualizationConfig(
            window_name="Knot Data Collection",
            window_width=self.resolution[0],
            window_height=self.resolution[1],
            show_info=True
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
            
    def _save_sample(self, frame: Frame, sample: KnotSample) -> Path:
        """Save a frame sample with metadata
        
        Args:
            frame: Frame to save
            sample: Sample metadata
            
        Returns:
            Path to the saved sample directory
        """
        # Create sample directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = self.base_path / sample.stage / timestamp
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RGB frame without overlays
        cv2.imwrite(str(sample_dir / "rgb.png"), frame.data)
        
        # Save metadata
        metadata_dict = {
            "stage": sample.stage,
            "capture_timestamp": sample.capture_timestamp,
            "notes": sample.notes,
            "frame_number": frame.frame_number,
            "resolution": f"{frame.width}x{frame.height}"
        }
        
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
            self.camera = GenericCamera(
                camera_index=self.camera_index,
                resolution=self.resolution,
                fps=self.fps
            )
            
            if not self.camera.initialize():
                print("Failed to initialize camera. Exiting.")
                return
            
            print(f"Camera initialized successfully (using index {self.camera_index})")
            
            print("\nOverhand Knot Data Collection")
            print("============================")
            print("\nCurrent sample counts:")
            for stage in self.STAGES:
                print(f"  {stage}: {self.sample_counts[stage]}")
            
            print("\nControls:")
            print("  'space': Capture sample")
            print("  's': Cycle knot stage")
            print("  'n': Add note to next capture")
            print("  'q': Quit")
            
            note_for_next = None
            
            while True:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # Show frame with current stage overlay
                    display_frame = frame.data.copy()
                    
                    # Add stage overlay
                    cv2.putText(display_frame, f"Stage: {self.current_stage}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Samples: {self.sample_counts[self.current_stage]}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (0, 255, 0), 2)
                    
                    if note_for_next:
                        cv2.putText(display_frame, f"Note: {note_for_next}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 165, 255), 2)
                              
                    # Add controls overlay
                    controls = [
                        "Controls:",
                        "SPACE: Capture sample",
                        "S: Cycle stage",
                        "N: Add note",
                        "Q: Quit"
                    ]
                    
                    y_pos = 120
                    for control in controls:
                        cv2.putText(display_frame, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (255, 255, 255), 2)  # White with outline
                        cv2.putText(display_frame, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 0), 1)
                        y_pos += 25
                    
                    # Show frame
                    key = self.viz.show(display_frame)
                    
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to capture
                        sample = KnotSample(
                            stage=self.current_stage,
                            capture_timestamp=datetime.now().isoformat(),
                            notes=note_for_next
                        )
                        
                        sample_dir = self._save_sample(frame, sample)
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
                        
                else:
                    # No frame available, just check for key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Small delay to avoid busy waiting
                    time.sleep(0.01)
                        
        finally:
            if self.camera:
                self.camera.close()
            if self.viz:
                self.viz.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect overhand knot RGB data')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Index of the camera to use (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width resolution (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height resolution (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera frames per second (default: 30)')
    parser.add_argument('--dataset-path', type=str, default="overhand_knot_dataset",
                       help='Path to dataset directory (default: overhand_knot_dataset)')
    args = parser.parse_args()
    
    collector = OverhandKnotCollector(
        camera_index=args.camera_index,
        base_path=args.dataset_path,
        resolution=(args.width, args.height),
        fps=args.fps
    )
    collector.start_collection()

if __name__ == "__main__":
    main()