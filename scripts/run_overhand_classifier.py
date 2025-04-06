#!/usr/bin/env python3
"""
Script to run the classifier API for the overhand knot
"""

import subprocess
import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the overhand knot classifier API')
    
    # Camera settings
    parser.add_argument('--camera-index', type=int, default=2,
                      help='Index of the camera to use (default: 2)')
    
    # Model settings
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                      help='Confidence threshold for classifications (default: 0.3)')
    
    # Sequential bias settings
    parser.add_argument('--sequential-bias', action='store_true',
                      help='Enable sequential bias in predictions')
    parser.add_argument('--bias-strength', type=float, default=1.0,
                      help='Strength of sequential bias effect (default: 1.0)')
    parser.add_argument('--bias-decay', type=float, default=0.5,
                      help='Decay rate for sequential bias effect (default: 0.5)')
    
    # Server settings
    parser.add_argument('--port', type=int, default=8000,
                      help='Port number for the server (default: 8000)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Knot-specific paths
    knot_name = "overhand_knot"
    model_path = f"models/{knot_name}/best_model.pth"
    knot_def_path = f"knot_definitions/{knot_name}.knot"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please run train_{knot_name}.py first to create the model.")
        sys.exit(1)
    
    # Check if knot definition exists
    if not Path(knot_def_path).exists():
        print(f"Error: Knot definition not found at {knot_def_path}")
        print(f"Please ensure knot_definitions directory contains {knot_name}.knot")
        sys.exit(1)
    
    # Run the API server
    print(f"Starting classifier API for {knot_name}...")
    print(f"Model: {model_path}")
    print(f"Knot definition: {knot_def_path}")
    
    # Build command with sequential bias if enabled
    cmd = [
        "python", "-m", "knots.classifier_api",
        "--model-path", model_path,
        "--camera-index", str(args.camera_index),
        "--knot-def-path", knot_def_path,
        "--confidence-threshold", str(args.confidence_threshold),
        "--port", str(args.port)
    ]
    
    # Add sequential bias options if enabled
    if args.sequential_bias:
        cmd.append("--sequential-bias")
        cmd.extend(["--bias-strength", str(args.bias_strength)])
        cmd.extend(["--bias-decay", str(args.bias_decay)])
        print(f"Sequential bias enabled:")
        print(f"  - Strength: {args.bias_strength}")
        print(f"  - Decay: {args.bias_decay}")
    else:
        print("Sequential bias disabled")
    
    try:
        # Print access instructions
        print("\nAPI will be available at:")
        print(f"  - Web interface: http://localhost:{args.port}/docs")
        print(f"  - Video stream: http://localhost:{args.port}/stream")
        print(f"  - Classification data: http://localhost:{args.port}/classification")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run the server process
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping server...")
    
if __name__ == "__main__":
    main()
