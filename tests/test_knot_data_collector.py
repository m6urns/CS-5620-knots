import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from knots.knot_definition import KnotDefinition, KnotStage
from knots.knot_data_collector import KnotDataCollector, KnotSample

# Mock Frame class for testing
@pytest.fixture
def mock_frame():
    class MockFrame:
        def __init__(self):
            self.data = 255 * MagicMock()  # Just a mock numpy array
            self.timestamp = int(datetime.now().timestamp() * 1_000_000)
            self.frame_number = 1
            self.width = 640
            self.height = 480
    
    return MockFrame()

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

@pytest.fixture
def sample_knot_def_file(sample_knot_def, tmp_path):
    """Create a sample knot definition file for testing"""
    file_path = tmp_path / "test_knot.knot"
    sample_knot_def.to_file(file_path)
    return file_path

@pytest.fixture
def mock_camera():
    """Create a mock camera for testing"""
    camera = MagicMock()
    camera.initialize.return_value = True
    return camera

# Test core functionality
def test_init_with_knot_def_file(sample_knot_def_file, tmp_path):
    """Test initializing with a knot definition file"""
    with patch('knots.knot_data_collector.GenericCamera'):
        collector = KnotDataCollector(
            knot_def_path=str(sample_knot_def_file),
            base_path=tmp_path / "test_dataset"
        )
        
        # Verify knot definition was loaded
        assert collector.knot_def is not None
        assert collector.knot_def.name == "test_knot"
        assert len(collector.knot_def.stages) == 3
        assert collector.knot_def.stage_ids == ["stage1", "stage2", "stage3"]
        
        # Verify base path was set correctly
        assert collector.base_path == tmp_path / "test_dataset"
        
        # Verify knot definition was saved in the dataset directory
        assert (tmp_path / "test_dataset" / "test_knot.knot").exists()
        
        # Verify sample counts were initialized
        assert collector.sample_counts == {"stage1": 0, "stage2": 0, "stage3": 0}
        
        # Verify current stage was set to first stage
        assert collector.current_stage == "stage1"
        assert collector.current_stage_idx == 0

def test_init_without_knot_def_file(tmp_path):
    """Test initializing without a knot definition file (default behavior)"""
    with patch('knots.knot_data_collector.GenericCamera'):
        collector = KnotDataCollector(
            base_path=tmp_path / "default_dataset"
        )
        
        # Verify default knot definition was created
        assert collector.knot_def is not None
        assert collector.knot_def.name == "overhand_knot"
        assert len(collector.knot_def.stages) == 4
        assert collector.knot_def.stage_ids == ["loose", "loop", "complete", "tightened"]
        
        # Verify base path was set correctly
        assert collector.base_path == tmp_path / "default_dataset"
        
        # Verify knot definition was saved in the dataset directory
        assert (tmp_path / "default_dataset" / "overhand_knot.knot").exists()
        
        # Verify sample counts were initialized
        assert collector.sample_counts == {
            "loose": 0, "loop": 0, "complete": 0, "tightened": 0
        }
        
        # Verify current stage was set to first stage
        assert collector.current_stage == "loose"
        assert collector.current_stage_idx == 0

def test_save_sample(mock_frame, tmp_path):
    """Test saving a sample"""
    with patch('knots.knot_data_collector.GenericCamera'), \
         patch('knots.knot_data_collector.cv2.imwrite') as mock_imwrite:
        collector = KnotDataCollector(
            base_path=tmp_path / "save_sample_test"
        )
        
        # Create a test sample
        sample = KnotSample(
            stage="loose",
            capture_timestamp=datetime.now().isoformat(),
            knot_type="overhand_knot",
            notes="Test note"
        )
        
        # Save the sample
        sample_dir = collector._save_sample(mock_frame, sample)
        
        # Verify imwrite was called to save the frame
        mock_imwrite.assert_called_once()
        
        # Verify metadata.json was created with correct content
        metadata_path = sample_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        assert metadata["stage"] == "loose"
        assert metadata["knot_type"] == "overhand_knot"
        assert metadata["notes"] == "Test note"
        assert metadata["stage_name"] == "Loose Rope"
        assert "resolution" in metadata
        
        # Verify sample counts were updated and saved
        assert collector.sample_counts["loose"] == 1
        assert collector.counts_path.exists()
        
        with open(collector.counts_path, 'r') as f:
            counts = json.load(f)
            
        assert counts["loose"] == 1

def test_cycle_stage(tmp_path):
    """Test cycling through stages"""
    with patch('knots.knot_data_collector.GenericCamera'):
        collector = KnotDataCollector(
            base_path=tmp_path / "cycle_stage_test"
        )
        
        # Verify initial stage
        assert collector.current_stage == "loose"
        assert collector.current_stage_idx == 0
        
        # Cycle stage
        original_idx = collector.current_stage_idx
        original_stage = collector.current_stage
        
        # Simulate pressing 's' by directly calling the cycle logic
        collector.current_stage_idx = (collector.current_stage_idx + 1) % len(collector.knot_def.stages)
        collector.current_stage = collector.knot_def.stage_id_from_index(collector.current_stage_idx)
        
        # Verify stage has changed
        assert collector.current_stage_idx != original_idx
        assert collector.current_stage != original_stage
        assert collector.current_stage == "loop"
        assert collector.current_stage_idx == 1
        
        # Cycle again
        collector.current_stage_idx = (collector.current_stage_idx + 1) % len(collector.knot_def.stages)
        collector.current_stage = collector.knot_def.stage_id_from_index(collector.current_stage_idx)
        
        assert collector.current_stage == "complete"
        assert collector.current_stage_idx == 2
        
        # Cycle to last stage
        collector.current_stage_idx = (collector.current_stage_idx + 1) % len(collector.knot_def.stages)
        collector.current_stage = collector.knot_def.stage_id_from_index(collector.current_stage_idx)
        
        assert collector.current_stage == "tightened"
        assert collector.current_stage_idx == 3
        
        # Cycle back to first stage
        collector.current_stage_idx = (collector.current_stage_idx + 1) % len(collector.knot_def.stages)
        collector.current_stage = collector.knot_def.stage_id_from_index(collector.current_stage_idx)
        
        assert collector.current_stage == "loose"
        assert collector.current_stage_idx == 0

def test_text_wrapping():
    """Test text wrapping function"""
    with patch('knots.knot_data_collector.GenericCamera'):
        collector = KnotDataCollector()
        
        # Test empty text
        assert collector._wrap_text("", 10) == [""]
        
        # Test short text
        assert collector._wrap_text("Short text", 20) == ["Short text"]
        
        # Test text that needs wrapping
        text = "This is a long text that needs to be wrapped to multiple lines"
        wrapped = collector._wrap_text(text, 20)
        
        assert len(wrapped) > 1
        assert wrapped[0] == "This is a long text"
        assert wrapped[1].startswith("that needs")
        
        # Verify the wrapped text has no lines longer than max_width
        for line in wrapped:
            assert len(line) <= 20

def test_load_existing_counts(tmp_path):
    """Test loading existing sample counts"""
    # Create existing counts file
    counts_dir = tmp_path / "existing_counts"
    counts_dir.mkdir(parents=True, exist_ok=True)
    counts_path = counts_dir / "sample_counts.json"
    
    with open(counts_path, 'w') as f:
        json.dump({
            "loose": 5,
            "loop": 3
        }, f)
    
    with patch('knots.knot_data_collector.GenericCamera'):
        collector = KnotDataCollector(
            base_path=counts_dir
        )
        
        # Verify existing counts were loaded and missing stages were initialized
        assert collector.sample_counts["loose"] == 5
        assert collector.sample_counts["loop"] == 3
        assert collector.sample_counts["complete"] == 0
        assert collector.sample_counts["tightened"] == 0