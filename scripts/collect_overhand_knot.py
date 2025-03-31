#!/usr/bin/env python3
"""
Example script for collecting overhand knot data
"""

import os
from pathlib import Path
from knots.knot_data_collector import KnotDataCollector

def main():
    # Create directory structure if it doesn't exist
    knot_name = "overhand_knot"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize collector with the overhand knot definition
    collector = KnotDataCollector(
        knot_def_path="knot_definitions/overhand_knot.knot",
        camera_index=2,  # Use the default camera
        base_path=f"data/{knot_name}",
        resolution=(640, 480),
        fps=30
    )
    
    # Start interactive data collection
    collector.start_collection()

if __name__ == "__main__":
    main()
