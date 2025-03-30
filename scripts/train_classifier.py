import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from knots.knot_definition import KnotDefinition
from knots.knot_classifier import KnotClassifier, KnotDataset, train_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train knot classifier')
    
    # Dataset arguments
    parser.add_argument('--data-path', type=str, default="overhand_knot_dataset",
                      help='Path to dataset directory (default: overhand_knot_dataset)')
    parser.add_argument('--knot-def-path', type=str, default=None,
                      help='Path to knot definition file (.knot) (default: None, will try to load from dataset)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training (default: 4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Validation split ratio (default: 0.2)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs (default: 30)')
    parser.add_argument('--unfreeze-epoch', type=int, default=10,
                      help='Epoch to unfreeze backbone layers (default: 10)')
    parser.add_argument('--early-stopping', type=int, default=5,
                      help='Patience for early stopping (default: 5)')
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA even if available')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default="models",
                      help='Directory to save model and results (default: models)')
    
    return parser.parse_args()

def load_knot_definition(knot_def_path, data_path):
    """Load knot definition from file or try to find in dataset directory
    
    Args:
        knot_def_path: Path to knot definition file or None
        data_path: Path to dataset directory (to search for knot definition)
        
    Returns:
        KnotDefinition object or None if not found
    """
    # Load knot definition if provided
    knot_def = None
    if knot_def_path:
        try:
            knot_def = KnotDefinition.from_file(knot_def_path)
            print(f"Loaded knot definition: {knot_def.name}")
            print(f"Stages: {knot_def.stage_ids}")
            print(f"Description: {knot_def.description}")
        except Exception as e:
            print(f"Error loading knot definition: {e}")
    else:
        # Try to find knot definition in the dataset directory
        knot_files = list(Path(data_path).glob("*.knot"))
        if knot_files:
            try:
                knot_def = KnotDefinition.from_file(knot_files[0])
                print(f"Found and loaded knot definition from dataset: {knot_def.name}")
            except Exception as e:
                print(f"Error loading knot definition from dataset: {e}")
    
    return knot_def

def create_data_loaders(data_path, knot_def, batch_size, val_split):
    """Create training and validation data loaders
    
    Args:
        data_path: Path to dataset directory
        knot_def: Knot definition or None
        batch_size: Batch size for training
        val_split: Validation split ratio (0.0 to 1.0)
        
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    print(f"Loading dataset from {data_path}...")
    
    # Create dataset with knot definition
    dataset = KnotDataset(data_path, knot_def=knot_def)
    
    # Get class names and distribution
    class_names = dataset.class_names
    distribution = dataset.class_distribution
    
    # Calculate split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    print("\nClass distribution:")
    for stage, count in distribution.items():
        print(f"  {stage}: {count} samples")
    
    return train_loader, val_loader, class_names

def evaluate_model(model, val_loader, device, class_names, output_dir):
    """Evaluate model and generate confusion matrix
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to evaluate on
        class_names: List of class names
        output_dir: Directory to save results
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(rgb)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, output_dict=True)
    
    # Print report summary
    print("\nClassification Report:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nClass metrics:")
    for cls in class_names:
        print(f"  {cls}: Precision={report[cls]['precision']:.4f}, "
              f"Recall={report[cls]['recall']:.4f}, "
              f"F1-score={report[cls]['f1-score']:.4f}")
    
    return report

def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load knot definition
    knot_def = load_knot_definition(args.knot_def_path, args.data_path)
    
    # Determine knot name for output directory
    knot_name = "generic_knot"
    if knot_def:
        knot_name = knot_def.name
    
    # Create knot-specific output directory
    output_dir = os.path.join(args.output_dir, knot_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_path, knot_def, args.batch_size, args.val_split)
    
    # Create model
    print("\nInitializing model...")
    model = KnotClassifier(knot_def=knot_def)
    model = model.to(device)
    
    # Train model
    print("\nStarting training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update train_model function in the actual implementation to return best model
    # For now, we'll just use the returned model from training
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        unfreeze_epoch=args.unfreeze_epoch,
        early_stopping_patience=args.early_stopping
    )
    
    # Save the best model directly to the knot-specific output directory
    best_model_path = os.path.join(output_dir, f'best_model.pth')
    
    # Save with knot definition if available
    if hasattr(model, 'knot_def') and model.knot_def is not None:
        model.save_with_knot_def(best_model_path)
    else:
        torch.save(model.state_dict(), best_model_path)
        
    print(f"\nBest model saved to {best_model_path}")
    
    # Also save a timestamped version for tracking
    timestamped_model_path = os.path.join(output_dir, f'model_{timestamp}.pth')
    if hasattr(model, 'knot_def') and model.knot_def is not None:
        model.save_with_knot_def(timestamped_model_path)
    else:
        torch.save(model.state_dict(), timestamped_model_path)
    print(f"Timestamped model saved to {timestamped_model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = evaluate_model(model, val_loader, device, class_names, output_dir)
    
    # Save evaluation metrics to a JSON file
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()