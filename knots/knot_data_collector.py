import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

from knots.knot_definition import KnotDefinition, KnotStage
from knots.utils.generic_camera import GenericCamera, Frame
from knots.utils.visualization import FrameVisualizer, VisualizationConfig

@dataclass
class KnotSample:
    """Metadata for a knot sample"""
    stage: str  # ID of the stage (e.g., "loose", "loop")
    capture_timestamp: str
    knot_type: str  # Type of knot (e.g., "overhand_knot")
    notes: Optional[str] = None

class KnotDataCollector:
    """Tool for collecting knot RGB data based on a knot definition"""
    
    def __init__(self, knot_def_path: Optional[str] = None, camera_index: int = 0, 
                 base_path: Optional[str] = None, resolution: tuple = (640, 480), 
                 fps: int = 30):
        """Initialize data collector
        
        Args:
            knot_def_path: Path to knot definition file (optional)
            camera_index: Index of the camera to use
            base_path: Base directory for dataset storage (defaults to knot_name_dataset)
            resolution: Camera resolution as (width, height)
            fps: Camera frames per second
        """
        # Load knot definition if provided
        self.knot_def = None
        if knot_def_path:
            try:
                self.knot_def = KnotDefinition.from_file(knot_def_path)
                print(f"Loaded knot definition: {self.knot_def.name}")
                print(f"Description: {self.knot_def.description}")
                print(f"Stages: {len(self.knot_def.stages)}")
                for stage in self.knot_def.stages:
                    print(f"  - {stage.name} ({stage.id}): {stage.description}")
            except Exception as e:
                print(f"Error loading knot definition: {str(e)}")
                print("Using default overhand knot stages")
                self.knot_def = self._create_default_knot_def()
        else:
            # Use default overhand knot definition for backward compatibility
            self.knot_def = self._create_default_knot_def()
        
        # Setup base path with the new structure
        if base_path is None:
            # Ensure data directory exists
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            base_path = f"data/{self.knot_def.name}_dataset"
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Save knot definition in the dataset directory
        self._save_knot_definition()
        
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        
        # Setup visualization with info overlay
        self.viz_config = VisualizationConfig(
            window_name=f"Knot Data Collection: {self.knot_def.name}",
            window_width=self.resolution[0],
            window_height=self.resolution[1],
            show_info=True
        )
        self.viz = FrameVisualizer(self.viz_config)
        
        # Initialize camera
        self.camera = None
        
        # Initialize current stage - start with the first stage
        self.current_stage_idx = 0
        self.current_stage = self.knot_def.stage_ids[0]
        
        # Initialize or load sample counts
        self.counts_path = self.base_path / "sample_counts.json"
        self.sample_counts = self._load_counts()
        
    def _create_default_knot_def(self) -> KnotDefinition:
        """Create default overhand knot definition for backward compatibility"""
        stages = [
            KnotStage("loose", "Loose Rope", "Starting position with loose rope"),
            KnotStage("loop", "Loop Created", "A loop has been formed but not pulled through"),
            KnotStage("complete", "Knot Completed", "Basic knot structure is complete but not tightened"),
            KnotStage("tightened", "Knot Tightened", "The knot has been tightened and is secure")
        ]
        
        return KnotDefinition(
            name="overhand_knot",
            description="A basic overhand knot with four stages",
            stages=stages
        )
    
    def _save_knot_definition(self):
        """Save the current knot definition in the dataset directory"""
        if self.knot_def:
            knot_def_path = self.base_path / f"{self.knot_def.name}.knot"
            self.knot_def.to_file(knot_def_path)
            print(f"Saved knot definition to {knot_def_path}")
        
    def _load_counts(self) -> Dict[str, int]:
        """Load or initialize sample counts for each stage"""
        if self.counts_path.exists():
            with open(self.counts_path, 'r') as f:
                counts = json.load(f)
                
            # Ensure all stages have a count
            for stage_id in self.knot_def.stage_ids:
                if stage_id not in counts:
                    counts[stage_id] = 0
                    
            return counts
        else:
            # Initialize counts for all stages
            return {stage_id: 0 for stage_id in self.knot_def.stage_ids}
        
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
        
        # Get stage name and description
        stage = self.knot_def.get_stage(sample.stage)
        stage_name = stage.name
        stage_description = stage.description
        
        # Save metadata
        metadata_dict = {
            "knot_type": sample.knot_type,
            "stage": sample.stage,
            "stage_name": stage_name,
            "stage_description": stage_description,
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
            
            print(f"\n{self.knot_def.name} Data Collection")
            print("============================")
            print(f"\nDescription: {self.knot_def.description}")
            print("\nCurrent sample counts:")
            for stage_id in self.knot_def.stage_ids:
                stage = self.knot_def.get_stage(stage_id)
                print(f"  {stage.name} ({stage_id}): {self.sample_counts[stage_id]}")
            
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
                    
                    # Get current stage object
                    current_stage = self.knot_def.get_stage(self.current_stage)
                    
                    # Add stage overlay
                    cv2.putText(display_frame, f"Stage: {current_stage.name} ({self.current_stage})", 
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
                    
                    y_pos = 175
                    for control in controls:
                        cv2.putText(display_frame, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (255, 255, 255), 2)  # White with outline
                        cv2.putText(display_frame, control,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (0, 0, 0), 1)
                        y_pos += 25
                    
                    # Add stage description
                    description = current_stage.description
                    wrapped_description = self._wrap_text(description, 60)
                    
                    y_pos += 10  # Add some spacing
                    cv2.putText(display_frame, "Stage Description:",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 2)
                    cv2.putText(display_frame, "Stage Description:",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (0, 0, 0), 1)
                    y_pos += 25
                    
                    for line in wrapped_description:
                        cv2.putText(display_frame, line,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)
                        cv2.putText(display_frame, line,
                                  (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (0, 0, 0), 1)
                        y_pos += 20
                    
                    # Show frame
                    key = self.viz.show(display_frame)
                    
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Space to capture
                        sample = KnotSample(
                            stage=self.current_stage,
                            capture_timestamp=datetime.now().isoformat(),
                            knot_type=self.knot_def.name,
                            notes=note_for_next
                        )
                        
                        sample_dir = self._save_sample(frame, sample)
                        print(f"\nSaved {current_stage.name} sample to {sample_dir}")
                        print(f"Total {current_stage.name} samples: {self.sample_counts[self.current_stage]}")
                        note_for_next = None
                        
                    elif key == ord('s'):
                        # Cycle stage
                        self.current_stage_idx = (self.current_stage_idx + 1) % len(self.knot_def.stages)
                        self.current_stage = self.knot_def.stage_id_from_index(self.current_stage_idx)
                        current_stage = self.knot_def.get_stage(self.current_stage)
                        print(f"\nStage: {current_stage.name} ({self.current_stage})")
                        print(f"Description: {current_stage.description}")
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
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width characters"""
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            # Check if adding this word would exceed max width
            if current_width + len(word) + (1 if current_width > 0 else 0) > max_width:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = len(word)
            else:
                if current_width > 0:
                    current_width += 1  # Add space
                current_line.append(word)
                current_width += len(word)
        
        # Add the last line
        if current_line:
            lines.append(" ".join(current_line))
            
        return lines

def main():
    """Main entry point with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect knot RGB data')
    parser.add_argument('--knot-def-path', type=str, default=None,
                       help='Path to knot definition file (.knot)')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Index of the camera to use (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width resolution (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height resolution (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera frames per second (default: 30)')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to dataset directory (default: knot_name_dataset)')
    args = parser.parse_args()
    
    collector = KnotDataCollector(
        knot_def_path=args.knot_def_path,
        camera_index=args.camera_index,
        base_path=args.dataset_path,
        resolution=(args.width, args.height),
        fps=args.fps
    )
    collector.start_collection()

if __name__ == "__main__":
    main()