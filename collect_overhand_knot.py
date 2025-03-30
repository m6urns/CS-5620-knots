#!/usr/bin/env python3
"""
Example script for collecting overhand knot data
"""

from knots.knot_data_collector import KnotDataCollector

def main():
    # Initialize collector with the overhand knot definition
    collector = KnotDataCollector(
        knot_def_path="knot_definitions/overhand_knot.knot",
        camera_index=0,  # Use the default camera
        base_path="overhand_knot_dataset",  # Custom dataset path
        resolution=(640, 480),
        fps=30
    )
    
    # Start interactive data collection
    collector.start_collection()

if __name__ == "__main__":
    main()