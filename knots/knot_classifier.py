import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, Any, List, Union

from knots.knot_definition import KnotDefinition

class KnotClassifier(nn.Module):
    """RGB-only classifier for knot stages"""
    
    def __init__(self, num_classes: int = 4, weights: str = 'DEFAULT', 
                 knot_def: Optional[KnotDefinition] = None):
        """Initialize RGB-only classifier
        
        Args:
            num_classes: Number of knot stages to classify (ignored if knot_def is provided)
            weights: Pretrained weights option for backbone ('DEFAULT', None, etc.)
            knot_def: Optional knot definition to use for setting up the classifier
        """
        # If knot_def is provided, get the number of classes from it
        if knot_def is not None:
            num_classes = knot_def.stage_count
            
        super().__init__()
        
        # Store knot definition if provided
        self.knot_def = knot_def
        
        # RGB backbone - Using EfficientNet-B0 for good performance/size trade-off
        self.backbone = models.efficientnet_b2(weights=weights)
        
        # Get feature dimensions - fix for different PyTorch versions
        if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
            # For older PyTorch versions where classifier is a Sequential
            features = self.backbone.classifier[-1].in_features
        elif hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
            # For versions where classifier is a single Linear layer
            features = self.backbone.classifier.in_features
        else:
            # Default for EfficientNet B0
            features = 1280
            
        # Replace classifier with Identity
        self.backbone.classifier = nn.Identity()
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone layers initially
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone layers to speed up initial training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Forward pass using only RGB input
        
        Args:
            rgb: RGB tensor of shape [batch_size, 3, H, W]
            
        Returns:
            Tensor of logits for each class
        """
        features = self.backbone(rgb)
        return self.classifier(features)
    
    @property
    def stage_ids(self) -> List[str]:
        """Get the list of stage IDs if knot definition is available"""
        if self.knot_def is not None:
            return self.knot_def.stage_ids
        return []
    
    def save_with_knot_def(self, path: Union[str, Path]):
        """Save the model along with the knot definition
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        
        # Save model state
        torch.save({
            'state_dict': self.state_dict(),
            'knot_def': self.knot_def.to_dict() if self.knot_def else None
        }, path)
    
    @classmethod
    def load_with_knot_def(cls, path: Union[str, Path], device: Optional[torch.device] = None):
        """Load the model along with the knot definition
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded KnotClassifier with knot definition
        """
        path = Path(path)
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved data
        saved_data = torch.load(path, map_location=device)
        
        # Recreate knot definition if available
        knot_def = None
        if saved_data.get('knot_def'):
            knot_def = KnotDefinition.from_dict(saved_data['knot_def'])
        
        # Create model
        model = cls(knot_def=knot_def)
        model.load_state_dict(saved_data['state_dict'])
        model.to(device)
        
        return model

class KnotDataset(Dataset):
    """Dataset for knot RGB data"""
    
    def __init__(self, data_path: str, transform: Optional[Any] = None,
                 knot_def: Optional[KnotDefinition] = None):
        """Initialize dataset
        
        Args:
            data_path: Path to dataset directory
            transform: Optional transform to apply to RGB images
            knot_def: Optional knot definition
        """
        self.data_path = Path(data_path)
        self.transform = transform or self._default_transform()
        self.knot_def = knot_def
        self.samples = self._load_samples()
    
    def _default_transform(self):
        """Default RGB data transformations - based on transforms from https://github.com/joecameron1/individualproject"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(360),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=0,  # No additional rotation here since we have RandomRotation
                translate=(0.2, 0.2),  # Width/height shift
                shear=20,  # Shear transformation
                scale=(0.7, 1.3),  # Zoom range equivalent to zoom_range=0.3
            ),
            # Keep color jittering
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        """Load all samples from the dataset"""
        samples = []
        
        # Determine stages from knot definition or directory structure
        if self.knot_def is not None:
            stages = self.knot_def.stage_ids
        else:
            # Try to infer stages from directory structure
            stages = []
            for item in self.data_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    stages.append(item.name)
            
            if not stages:
                raise ValueError(f"No stage directories found in {self.data_path}")
            
            # Try to load knot definition from dataset directory
            knot_files = list(self.data_path.glob("*.knot"))
            if knot_files:
                try:
                    self.knot_def = KnotDefinition.from_file(knot_files[0])
                    print(f"Loaded knot definition from {knot_files[0]}")
                    # Update stages to match the loaded knot definition
                    stages = [stage for stage in self.knot_def.stage_ids if stage in stages]
                except Exception as e:
                    print(f"Failed to load knot definition: {e}")
        
        # Track the mapping from stage ID to label index
        self.stage_to_idx = {stage: idx for idx, stage in enumerate(stages)}
        
        # Load samples for each stage
        for stage_id in stages:
            stage_path = self.data_path / stage_id
            if not stage_path.exists():
                print(f"Warning: Stage directory '{stage_id}' not found in dataset")
                continue
                
            for sample_dir in stage_path.iterdir():
                if not sample_dir.is_dir():
                    continue
                    
                rgb_path = sample_dir / "rgb.png"
                metadata_path = sample_dir / "metadata.json"
                
                if rgb_path.exists():
                    samples.append({
                        'rgb_path': str(rgb_path),
                        'metadata_path': str(metadata_path) if metadata_path.exists() else None,
                        'stage': stage_id,
                        'label': self.stage_to_idx[stage_id]
                    })
        
        print(f"Loaded {len(samples)} samples across {len(stages)} stages")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        rgb = cv2.imread(sample['rgb_path'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            rgb = self.transform(rgb)
        
        return {
            'rgb': rgb,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }
    
    @property
    def class_names(self):
        """Get class names based on stage IDs"""
        if self.knot_def is not None:
            return self.knot_def.stage_names
        return list(self.stage_to_idx.keys())
    
    @property
    def class_distribution(self):
        """Get distribution of samples per class"""
        distribution = {stage: 0 for stage in self.stage_to_idx.keys()}
        for sample in self.samples:
            distribution[sample['stage']] += 1
        return distribution

def train_model(model, train_loader, val_loader, num_epochs=30, device='cpu',
             unfreeze_epoch=10, early_stopping_patience=5, lr=0.001):
    """Train the knot classifier model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs to train for
        device: Device to train on ('cpu' or 'cuda')
        unfreeze_epoch: Epoch to unfreeze backbone layers (for transfer learning)
        early_stopping_patience: Number of epochs to wait before early stopping
        lr: Learning rate
        
    Returns:
        The trained model (best performing model on validation set)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    import copy
    import time
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Initially, only train the classifier head
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable training for classifier layers
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Training metrics
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Unfreeze backbone after specified epoch
        if epoch == unfreeze_epoch:
            print("Unfreezing backbone layers...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer to include all parameters
            optimizer = optim.Adam(model.parameters(), lr=lr/10)
            print(f"Adjusted learning rate to {lr/10}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch in train_loader:
            inputs = batch['rgb'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['rgb'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Check if this is the best model so far
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best model with validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Training Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model

# Simple test script
def main():
    """Test script for the classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train knot classifier')
    parser.add_argument('--data-path', type=str, default="overhand_knot_dataset",
                      help='Path to dataset')
    parser.add_argument('--knot-def-path', type=str, default=None, 
                      help='Path to knot definition file (.knot)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    
    args = parser.parse_args()
    
    # Load knot definition if provided
    knot_def = None
    if args.knot_def_path:
        try:
            knot_def = KnotDefinition.from_file(args.knot_def_path)
            print(f"Loaded knot definition: {knot_def.name}")
            print(f"Stages: {knot_def.stage_ids}")
        except Exception as e:
            print(f"Error loading knot definition: {e}")
    
    # Setup dataset
    dataset = KnotDataset(args.data_path, knot_def=knot_def)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"Dataset sizes:")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    print("\nClass distribution:")
    for stage, count in dataset.class_distribution.items():
        print(f"  {stage}: {count} samples")
        
    # Create and train model
    model = KnotClassifier(knot_def=knot_def)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = train_model(model, train_loader, val_loader, device=device)

if __name__ == '__main__':
    main()