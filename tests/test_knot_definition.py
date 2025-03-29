import os
import json
import pytest
from pathlib import Path

from knots.knot_definition import KnotDefinition, KnotStage

@pytest.fixture
def sample_overhand_stages():
    """Fixture to provide sample stages for an overhand knot"""
    return [
        KnotStage("loose", "Loose Rope", "Starting position with loose rope"),
        KnotStage("loop", "Loop Created", "A loop has been formed but not pulled through"),
        KnotStage("complete", "Knot Completed", "Basic knot structure is complete but not tightened"),
        KnotStage("tightened", "Knot Tightened", "The knot has been tightened and is secure")
    ]

@pytest.fixture
def sample_overhand_knot(sample_overhand_stages, tmp_path):
    """Fixture to provide a sample overhand knot definition and save it to a file"""
    knot_def = KnotDefinition(
        name="overhand_knot",
        description="A basic overhand knot with four stages",
        stages=sample_overhand_stages
    )
    
    # Save to a temporary file
    file_path = tmp_path / "overhand_knot.knot"
    knot_def.to_file(file_path)
    
    return {
        "knot_def": knot_def,
        "file_path": file_path
    }

@pytest.fixture
def figure_eight_knot_data():
    """Fixture to provide data for a figure-eight knot"""
    return {
        "name": "figure_eight_knot",
        "description": "A figure-eight knot with seven stages",
        "stages": [
            {"id": "starting", "name": "Starting Position", "description": "Loose rope in initial position"},
            {"id": "first_loop", "name": "First Loop", "description": "Create the first loop (away from the end)"},
            {"id": "pass_through", "name": "Pass Through", "description": "Pass the working end up through the loop"},
            {"id": "wrap_around", "name": "Wrap Around", "description": "Wrap the working end around the standing part"},
            {"id": "pass_back", "name": "Pass Back", "description": "Pass the working end back down through the first loop"},
            {"id": "tighten", "name": "Tighten", "description": "Pull all ends to tighten the knot"},
            {"id": "complete", "name": "Complete", "description": "A tightened, completed figure-eight knot"}
        ]
    }

@pytest.fixture
def square_knot_data():
    """Fixture to provide data for a square knot"""
    return {
        "name": "square_knot",
        "description": "A square knot (reef knot) with eight stages",
        "stages": [
            {"id": "starting", "name": "Starting Position", "description": "Two rope ends side by side"},
            {"id": "right_over_left", "name": "Right Over Left", "description": "Cross right end over left"},
            {"id": "right_under_left", "name": "Right Under Left", "description": "Pass right end under left"},
            {"id": "half_complete", "name": "Half-Knot Complete", "description": "First half of the knot is complete"},
            {"id": "left_over_right", "name": "Left Over Right", "description": "Cross left end over right"},
            {"id": "left_under_right", "name": "Left Under Right", "description": "Pass left end under right"},
            {"id": "tighten", "name": "Tighten", "description": "Pull all ends to tighten the knot"},
            {"id": "complete", "name": "Complete", "description": "A tightened, completed square knot"}
        ]
    }

# Core functionality tests
def test_knot_stage_creation():
    """Test creation of a knot stage"""
    stage = KnotStage("test_id", "Test Name", "Test description")
    
    assert stage.id == "test_id"
    assert stage.name == "Test Name"
    assert stage.description == "Test description"
    assert str(stage) == "Test Name (test_id)"

def test_knot_stage_empty_name():
    """Test that an empty name defaults to the ID"""
    stage = KnotStage("test_id", "", "Test description")
    
    assert stage.id == "test_id"
    assert stage.name == "test_id"  # Name should default to ID
    assert stage.description == "Test description"

def test_knot_stage_empty_id():
    """Test that an empty ID raises an error"""
    with pytest.raises(ValueError, match="Stage ID cannot be empty"):
        KnotStage("", "Test Name", "Test description")

def test_knot_definition_creation(sample_overhand_stages):
    """Test creation of a knot definition"""
    knot_def = KnotDefinition(
        name="test_knot",
        description="Test description",
        stages=sample_overhand_stages
    )
    
    assert knot_def.name == "test_knot"
    assert knot_def.description == "Test description"
    assert len(knot_def.stages) == 4
    assert knot_def.stage_count == 4
    assert knot_def.stage_ids == ["loose", "loop", "complete", "tightened"]
    assert knot_def.stage_names == ["Loose Rope", "Loop Created", "Knot Completed", "Knot Tightened"]

def test_knot_definition_empty_name(sample_overhand_stages):
    """Test that an empty name raises an error"""
    with pytest.raises(ValueError, match="Knot name cannot be empty"):
        KnotDefinition("", "Test description", sample_overhand_stages)

def test_knot_definition_empty_stages():
    """Test that empty stages raises an error"""
    with pytest.raises(ValueError, match="Knot must have at least one stage"):
        KnotDefinition("test_knot", "Test description", [])

def test_knot_definition_duplicate_stage_ids():
    """Test that duplicate stage IDs raise an error"""
    stages = [
        KnotStage("stage1", "Stage 1", "Description 1"),
        KnotStage("stage2", "Stage 2", "Description 2"),
        KnotStage("stage1", "Duplicate ID", "This has the same ID as Stage 1")
    ]
    
    with pytest.raises(ValueError, match="Stage IDs must be unique"):
        KnotDefinition("Duplicate", "Knot with duplicate stage IDs", stages)

def test_knot_definition_utility_methods(sample_overhand_knot):
    """Test utility methods of KnotDefinition"""
    knot_def = sample_overhand_knot["knot_def"]
    
    # Test stage_index method
    assert knot_def.stage_index("loose") == 0
    assert knot_def.stage_index("loop") == 1
    assert knot_def.stage_index("complete") == 2
    assert knot_def.stage_index("tightened") == 3
    
    # Test stage_id_from_index method
    assert knot_def.stage_id_from_index(0) == "loose"
    assert knot_def.stage_id_from_index(1) == "loop"
    assert knot_def.stage_id_from_index(2) == "complete"
    assert knot_def.stage_id_from_index(3) == "tightened"
    
    # Test get_stage method
    stage = knot_def.get_stage("loop")
    assert stage.id == "loop"
    assert stage.name == "Loop Created"
    
    # Test get_stage_by_index method
    stage = knot_def.get_stage_by_index(2)
    assert stage.id == "complete"
    assert stage.name == "Knot Completed"

def test_knot_definition_error_handling(sample_overhand_knot):
    """Test error handling in KnotDefinition utility methods"""
    knot_def = sample_overhand_knot["knot_def"]
    
    # Test nonexistent stage ID
    with pytest.raises(ValueError, match="Stage ID 'nonexistent' not found"):
        knot_def.stage_index("nonexistent")
    
    with pytest.raises(ValueError, match="Stage ID 'nonexistent' not found"):
        knot_def.get_stage("nonexistent")
    
    # Test invalid stage index
    with pytest.raises(ValueError, match="Stage index 10 out of range"):
        knot_def.stage_id_from_index(10)
    
    with pytest.raises(ValueError, match="Stage index 10 out of range"):
        knot_def.get_stage_by_index(10)

# Serialization tests
def test_knot_stage_serialization():
    """Test serialization of KnotStage"""
    stage = KnotStage("test_id", "Test Name", "Test description")
    stage_dict = stage.to_dict()
    
    assert stage_dict == {
        "id": "test_id",
        "name": "Test Name",
        "description": "Test description"
    }
    
    # Test deserialization
    loaded_stage = KnotStage.from_dict(stage_dict)
    assert loaded_stage.id == stage.id
    assert loaded_stage.name == stage.name
    assert loaded_stage.description == stage.description

def test_knot_definition_serialization(sample_overhand_knot):
    """Test serialization of KnotDefinition"""
    knot_def = sample_overhand_knot["knot_def"]
    knot_dict = knot_def.to_dict()
    
    assert knot_dict["name"] == "overhand_knot"
    assert knot_dict["description"] == "A basic overhand knot with four stages"
    assert len(knot_dict["stages"]) == 4
    assert knot_dict["stages"][0]["id"] == "loose"
    
    # Test deserialization
    loaded_knot = KnotDefinition.from_dict(knot_dict)
    assert loaded_knot.name == knot_def.name
    assert loaded_knot.description == knot_def.description
    assert len(loaded_knot.stages) == len(knot_def.stages)
    assert loaded_knot.stage_ids == knot_def.stage_ids

# File operations tests
def test_knot_definition_save_and_load(sample_overhand_knot):
    """Test saving and loading a knot definition to/from a file"""
    knot_def = sample_overhand_knot["knot_def"]
    file_path = sample_overhand_knot["file_path"]
    
    # Verify file was created by the fixture
    assert file_path.exists()
    
    # Load from file
    loaded_knot = KnotDefinition.from_file(file_path)
    
    # Verify loaded knot matches original
    assert loaded_knot.name == knot_def.name
    assert loaded_knot.description == knot_def.description
    assert loaded_knot.stage_count == knot_def.stage_count
    assert loaded_knot.stage_ids == knot_def.stage_ids

def test_knot_definition_save_without_extension(tmp_path):
    """Test saving a knot definition without specifying an extension"""
    stages = [KnotStage("test", "Test Stage", "Test description")]
    knot_def = KnotDefinition("test_knot", "Test description", stages)
    
    # Save to file without extension
    file_path = tmp_path / "test_knot"
    knot_def.to_file(file_path)
    
    # Verify file was created with .knot extension
    assert (tmp_path / "test_knot.knot").exists()
    
    # Load the file and verify
    loaded_knot = KnotDefinition.from_file(tmp_path / "test_knot.knot")
    assert loaded_knot.name == "test_knot"

def test_knot_definition_file_not_found():
    """Test loading a nonexistent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="not found"):
        KnotDefinition.from_file("nonexistent_file.knot")

def test_knot_definition_invalid_json(tmp_path):
    """Test loading an invalid JSON file raises ValueError"""
    # Create invalid JSON file
    file_path = tmp_path / "invalid.knot"
    with open(file_path, 'w') as f:
        f.write("This is not valid JSON")
    
    with pytest.raises(ValueError, match="Invalid JSON"):
        KnotDefinition.from_file(file_path)

def test_knot_definition_missing_required_fields(tmp_path):
    """Test loading a file with missing required fields raises ValueError"""
    # Create file with missing fields
    file_path = tmp_path / "missing_fields.knot"
    with open(file_path, 'w') as f:
        json.dump({}, f)
    
    with pytest.raises(ValueError, match="missing required 'name' field"):
        KnotDefinition.from_file(file_path)

# Sample knot creation tests
def test_create_figure_eight_knot(tmp_path, figure_eight_knot_data):
    """Test creating a figure-eight knot definition"""
    # Create from dictionary data
    knot_def = KnotDefinition.from_dict(figure_eight_knot_data)
    
    # Verify basic properties
    assert knot_def.name == "figure_eight_knot"
    assert knot_def.stage_count == 7
    assert "first_loop" in knot_def.stage_ids
    
    # Save to file
    file_path = tmp_path / "figure_eight_knot.knot"
    knot_def.to_file(file_path)
    
    # Verify file exists
    assert file_path.exists()

def test_create_square_knot(tmp_path, square_knot_data):
    """Test creating a square knot definition"""
    # Create from dictionary data
    knot_def = KnotDefinition.from_dict(square_knot_data)
    
    # Verify basic properties
    assert knot_def.name == "square_knot"
    assert knot_def.stage_count == 8
    assert "right_over_left" in knot_def.stage_ids
    
    # Save to file
    file_path = tmp_path / "square_knot.knot"
    knot_def.to_file(file_path)
    
    # Verify file exists
    assert file_path.exists()

# Write example knot files to disk for later use
def test_write_example_knot_files(sample_overhand_stages, figure_eight_knot_data, square_knot_data):
    """Write example knot files to disk for use in the application"""
    # Create output directory if it doesn't exist
    output_dir = Path("knot_definitions")
    output_dir.mkdir(exist_ok=True)
    
    # Create overhand knot
    overhand_knot = KnotDefinition(
        name="overhand_knot",
        description="A basic overhand knot with four stages",
        stages=sample_overhand_stages
    )
    overhand_knot.to_file(output_dir / "overhand_knot.knot")
    
    # Create figure-eight knot
    figure_eight_knot = KnotDefinition.from_dict(figure_eight_knot_data)
    figure_eight_knot.to_file(output_dir / "figure_eight_knot.knot")
    
    # Create square knot
    square_knot = KnotDefinition.from_dict(square_knot_data)
    square_knot.to_file(output_dir / "square_knot.knot")
    
    # Verify files were created
    assert (output_dir / "overhand_knot.knot").exists()
    assert (output_dir / "figure_eight_knot.knot").exists()
    assert (output_dir / "square_knot.knot").exists()