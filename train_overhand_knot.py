#!/usr/bin/env python3
"""
Example script for training an overhand knot classifier
"""

import os
import sys
import torch
from pathlib import Path

from knots.knot_definition import KnotDefinition
from knots.knot_classifier import KnotClassifier, KnotDataset
from train_classifier import create_data_loaders, train_model, evaluate_model

def main():
    # Check that dataset exists
    data_path = "overhand_knot_dataset"
    if not Path(data_path).exists():
        print(f"Error: Dataset directory '{data_path}' not found.")
        print("Please run collect_overhand_data.py first to create the dataset.")
        sys.exit(1)
        
    # Load knot definition
    knot_def_path = "knot_definitions/overhand_knot.knot"
    try:
        knot_def = KnotDefinition.from_file(knot_def_path)
        print(f"Loaded knot definition: {knot_def.name}")
        print(f"Description: {knot_def.description}")
        print(f"Stages: {knot_def.stage_ids}")
    except Exception as e:
        print(f"Error loading knot definition: {e}")
        sys.exit(1)
    
    # Configure training parameters
    batch_size = 4
    val_split = 0.2
    epochs = 30
    unfreeze_epoch = 10
    early_stopping = 5
    output_dir = "overhand_knot_models"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        data_path, knot_def, batch_size, val_split)
    
    # Create model
    print(f"\nInitializing model for {knot_def.name}...")
    model = KnotClassifier(knot_def=knot_def)
    model = model.to(device)
    
    # Train model
    print("\nStarting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        device=device,
        unfreeze_epoch=unfreeze_epoch,
        early_stopping_patience=early_stopping
    )
    
    # Save the trained model
    model_path = os.path.join(output_dir, "overhand_knot_model.pth")
    model.save_with_knot_def(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device, class_names, output_dir)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()