#!/usr/bin/env python3
"""
Script to run the classifier API for the overhand knot
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Check if model exists
    model_path = "overhand_knot_models/overhand_knot_model.pth"
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_overhand_knot.py first to create the model.")
        sys.exit(1)
    
    # Check if knot definition exists
    knot_def_path = "knot_definitions/overhand_knot.knot"
    if not Path(knot_def_path).exists():
        print(f"Error: Knot definition not found at {knot_def_path}")
        print("Please ensure knot_definitions directory contains overhand_knot.knot")
        sys.exit(1)
    
    # Run the API server
    print(f"Starting classifier API for overhand knot...")
    print(f"Model: {model_path}")
    print(f"Knot definition: {knot_def_path}")
    
    cmd = [
        "python", "-m", "knots.classifier_api",
        "--model-path", model_path,
        "--knot-def-path", knot_def_path,
        "--confidence-threshold", "0.6",
        "--port", "8000"
    ]
    
    try:
        # Print access instructions
        print("\nAPI will be available at:")
        print("  - Web interface: http://localhost:8000/docs")
        print("  - Video stream: http://localhost:8000/stream")
        print("  - Classification data: http://localhost:8000/classification")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run the server process
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping server...")
    
if __name__ == "__main__":
    main()