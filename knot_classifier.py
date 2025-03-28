import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, Any

class KnotClassifier(nn.Module):
    """RGB-only classifier for knot stages"""
    
    def __init__(self, num_classes: int = 4, weights: str = 'DEFAULT'):
        """Initialize RGB-only classifier
        
        Args:
            num_classes: Number of knot stages to classify
            weights: Pretrained weights option for backbone ('DEFAULT', None, etc.)
        """
        super().__init__()
        
        # RGB backbone - Using EfficientNet-B0 for good performance/size trade-off
        self.backbone = models.efficientnet_b0(weights=weights)
        # Store feature dimension before replacing classifier
        features = self.backbone.classifier[1].in_features
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

class KnotDataset(Dataset):
    """Dataset for knot RGB data"""
    
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        """Initialize dataset
        
        Args:
            data_path: Path to dataset directory
            transform: Optional transform to apply to RGB images
        """
        self.data_path = Path(data_path)
        self.transform = transform or self._default_transform()
        self.samples = self._load_samples()
    
    def _default_transform(self):
        """Default RGB data transformations"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        """Load all samples from the dataset"""
        samples = []
        stages = ['loose', 'loop', 'complete', 'tightened']
        
        for stage_idx, stage in enumerate(stages):
            stage_path = self.data_path / stage
            if not stage_path.exists():
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
                        'stage': stage,
                        'label': stage_idx
                    })
        
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

def train_model(model, train_loader, val_loader, num_epochs=20, 
                device='cuda', unfreeze_epoch=10, early_stopping_patience=5):
    """Train the knot classifier model
    
    Args:
        model: KnotClassifier model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        unfreeze_epoch: Epoch after which to unfreeze backbone
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    model = model.to(device)
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Unfreeze backbone for fine-tuning after specified epoch
        if epoch == unfreeze_epoch:
            model._unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(rgb)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100 * correct / total
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f}, '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
            
    return model

# Simple test script
def main():
    """Test script for the classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train knot classifier')
    parser.add_argument('--data-path', type=str, default="overhand_knot_dataset",
                      help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    
    args = parser.parse_args()
    
    # Setup dataset
    dataset = KnotDataset(args.data_path)
    
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
    stage_counts = {}
    for sample in dataset.samples:
        stage = sample['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} samples")
        
    # Create and train model
    model = KnotClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = train_model(model, train_loader, val_loader, device=device)

if __name__ == '__main__':
    main()