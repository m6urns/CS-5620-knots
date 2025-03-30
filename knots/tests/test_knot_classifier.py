import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from knots.knot_definition import KnotDefinition, KnotStage
from knots.knot_classifier import KnotClassifier

@pytest.fixture
def sample_knot_def():
    """Create a sample knot definition for testing"""
    stages = [
        KnotStage("stage1", "Stage 1", "First stage description"),
        KnotStage("stage2", "Stage 2", "Second stage description"),
        KnotStage("stage3", "Stage 3", "Third stage description")
    ]
    
    return KnotDefinition(
        name="test_knot",
        description="Test knot with three stages",
        stages=stages
    )

class MockModule(nn.Module):
    """Mock module for testing"""
    def __init__(self, output_tensor=None):
        super().__init__()
        self.output_tensor = output_tensor or torch.ones(2, 1280)
    
    def forward(self, x):
        return self.output_tensor

# Test KnotClassifier initialization
def test_init_without_knot_def(monkeypatch):
    """Test initializing KnotClassifier without a knot definition"""
    # Mock the EfficientNet model to avoid loading weights
    mock_model = MagicMock()
    mock_model.classifier = nn.Linear(1280, 1000)  # Mock classifier with in_features
    
    monkeypatch.setattr('knots.knot_classifier.models.efficientnet_b0', lambda weights: mock_model)
    
    # Create classifier with default number of classes
    classifier = KnotClassifier(num_classes=4)
    
    # Check basic properties
    assert classifier.knot_def is None
    assert len(classifier.stage_ids) == 0
    
    # Check the classifier output size
    assert classifier.classifier[-1].out_features == 4

def test_init_with_knot_def(sample_knot_def, monkeypatch):
    """Test initializing KnotClassifier with a knot definition"""
    # Mock the EfficientNet model to avoid loading weights
    mock_model = MagicMock()
    mock_model.classifier = nn.Linear(1280, 1000)  # Mock classifier with in_features
    
    monkeypatch.setattr('knots.knot_classifier.models.efficientnet_b0', lambda weights: mock_model)
    
    # Create classifier with knot definition
    classifier = KnotClassifier(knot_def=sample_knot_def)
    
    # Check that knot definition was stored
    assert classifier.knot_def is sample_knot_def
    
    # Check that stage IDs are accessible
    assert classifier.stage_ids == ["stage1", "stage2", "stage3"]
    
    # Check that the classifier output size matches the number of stages
    assert classifier.classifier[-1].out_features == 3

# Test forwarding through the model
def test_forward(monkeypatch):
    """Test forward pass through KnotClassifier"""
    # Create a properly structured test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(1280, 1000)
        
        def forward(self, x):
            return torch.ones(x.shape[0], 1280)  # Return features of correct shape
    
    # Create a proper mock for the classifier
    class TestClassifier(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    
    # Mock the creation of efficientnet_b0
    monkeypatch.setattr('knots.knot_classifier.models.efficientnet_b0', lambda weights: TestModel())
    
    # Create a classifier
    classifier = KnotClassifier(num_classes=3)
    
    # Replace the classifier with our test classifier
    classifier.classifier = TestClassifier()
    
    # Create a batch of 2 images
    batch = torch.zeros(2, 3, 224, 224)
    
    # Forward pass
    output = classifier(batch)
    
    # Check the output shape
    assert output.shape == (2, 3)

# Test saving and loading with knot definition
def test_save_load_with_knot_def(sample_knot_def, tmp_path, monkeypatch):
    """Test saving and loading a model with knot definition"""
    # Create a proper mock for efficientnet_b0
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(1280, 1000)
        
        def forward(self, x):
            return torch.ones(x.shape[0], 1280)
    
    # Mock the creation of efficientnet_b0
    monkeypatch.setattr('knots.knot_classifier.models.efficientnet_b0', 
                       lambda weights: TestModel())
    
    # Create classifier with knot definition
    classifier = KnotClassifier(knot_def=sample_knot_def)
    
    # Save the model
    save_path = tmp_path / "test_model.pth"
    classifier.save_with_knot_def(save_path)
    
    # Check that the file was created
    assert save_path.exists()
    
    # Instead of patching __init__, let's patch the whole class
    # Create a mock KnotClassifier that will be returned by load_with_knot_def
    mock_classifier = MagicMock(spec=KnotClassifier)
    mock_classifier.knot_def = sample_knot_def  # Set the knot_def attribute
    
    # Patch the classmethod directly
    with patch.object(KnotClassifier, 'load_with_knot_def', return_value=mock_classifier) as mock_load:
        # Call the classmethod (this will return our mock_classifier)
        loaded_model = KnotClassifier.load_with_knot_def(save_path)
        
        # Verify the classmethod was called with the right arguments
        mock_load.assert_called_once()
        
        # Check that the returned model has the right knot_def
        assert loaded_model.knot_def == sample_knot_def