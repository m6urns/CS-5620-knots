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
from scripts.train_classifier import create_data_loaders, evaluate_model, train_model, load_knot_definition

def main():
    # Check that dataset exists
    data_path = "data/overhand_knot"
    if not Path(data_path).exists():
        print(f"Error: Dataset directory '{data_path}' not found.")
        print("Please run collect_overhand_data.py first to create the dataset.")
        sys.exit(1)
        
    # Load knot definition
    knot_def_path = "knot_definitions/overhand_knot.knot"
    knot_def = load_knot_definition(knot_def_path, data_path)
    if not knot_def:
        print("Failed to load knot definition")
        sys.exit(1)
    
    # Configure training parameters
    # batch_size = 4
    batch_size = 32
    val_split = 0.2
    # epochs = 50
    # unfreeze_epoch = 20
    epochs = 100
    unfreeze_epoch = 30
    early_stopping = 25
    learning_rate = 0.0001
    
    # Define knot-specific output directory
    output_dir = os.path.join("models", knot_def.name)
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
        early_stopping_patience=early_stopping,
        lr=learning_rate
    )
    
    # Save the trained model directly to the knot-specific output directory
    model_path = os.path.join(output_dir, "best_model.pth")
    model.save_with_knot_def(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device, class_names, output_dir)
    
    print("\nTraining complete!")
    print(f"\nTo run the classifier API, use: python run_overhand_classifier.py")

if __name__ == "__main__":
    main()