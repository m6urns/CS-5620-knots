import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

class KnotStage:
    """Represents a single stage in a knot tying process
    
    Attributes:
        id (str): Unique identifier for the stage
        name (str): Human-readable name for the stage
        description (str): Detailed description of the stage
    """
    
    def __init__(self, id: str, name: str, description: str):
        """Initialize a knot stage
        
        Args:
            id: Unique identifier for the stage
            name: Human-readable name for the stage
            description: Detailed description of the stage
        """
        if not id:
            raise ValueError("Stage ID cannot be empty")
        
        self.id = id
        self.name = name if name else id  # Default to ID if name not provided
        self.description = description
        
    def __str__(self) -> str:
        """Get string representation of the stage"""
        return f"{self.name} ({self.id})"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert stage to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnotStage':
        """Create stage from dictionary
        
        Args:
            data: Dictionary containing stage data
            
        Returns:
            KnotStage: Created stage instance
            
        Raises:
            ValueError: If data is missing required fields
        """
        if 'id' not in data:
            raise ValueError("Stage data missing required 'id' field")
        
        return cls(
            id=data['id'],
            name=data.get('name', data['id']),  # Default to ID if name not provided
            description=data.get('description', "")  # Default empty description
        )

class KnotDefinition:
    """Represents a complete knot definition with metadata and stages
    
    Attributes:
        name (str): Name of the knot
        description (str): Description of the knot
        stages (List[KnotStage]): Ordered list of knot stages
    """
    
    def __init__(self, name: str, description: str, stages: List[KnotStage]):
        """Initialize a knot definition
        
        Args:
            name: Name of the knot
            description: Description of the knot
            stages: Ordered list of knot stages
            
        Raises:
            ValueError: If name is empty or stages list is empty
        """
        if not name:
            raise ValueError("Knot name cannot be empty")
        if not stages:
            raise ValueError("Knot must have at least one stage")
        
        self.name = name
        self.description = description
        self.stages = stages
        
        # Verify stage IDs are unique
        stage_ids = [stage.id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Stage IDs must be unique within a knot definition")
        
    @property
    def stage_ids(self) -> List[str]:
        """Get list of stage IDs in order"""
        return [stage.id for stage in self.stages]
    
    @property
    def stage_names(self) -> List[str]:
        """Get list of stage names in order"""
        return [stage.name for stage in self.stages]
    
    @property
    def stage_count(self) -> int:
        """Get number of stages"""
        return len(self.stages)
    
    def stage_index(self, stage_id: str) -> int:
        """Get index of stage by ID
        
        Args:
            stage_id: ID of the stage to find
            
        Returns:
            int: Index of the stage
            
        Raises:
            ValueError: If stage ID is not found
        """
        try:
            return self.stage_ids.index(stage_id)
        except ValueError:
            raise ValueError(f"Stage ID '{stage_id}' not found in knot definition")
    
    def stage_id_from_index(self, index: int) -> str:
        """Get stage ID from index
        
        Args:
            index: Index of the stage
            
        Returns:
            str: ID of the stage
            
        Raises:
            ValueError: If index is out of range
        """
        if 0 <= index < len(self.stages):
            return self.stages[index].id
        raise ValueError(f"Stage index {index} out of range (0-{len(self.stages)-1})")
    
    def get_stage(self, stage_id: str) -> KnotStage:
        """Get stage by ID
        
        Args:
            stage_id: ID of the stage to get
            
        Returns:
            KnotStage: Stage with the given ID
            
        Raises:
            ValueError: If stage ID is not found
        """
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise ValueError(f"Stage ID '{stage_id}' not found in knot definition")
    
    def get_stage_by_index(self, index: int) -> KnotStage:
        """Get stage by index
        
        Args:
            index: Index of the stage
            
        Returns:
            KnotStage: Stage at the given index
            
        Raises:
            ValueError: If index is out of range
        """
        if 0 <= index < len(self.stages):
            return self.stages[index]
        raise ValueError(f"Stage index {index} out of range (0-{len(self.stages)-1})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knot definition to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'stages': [stage.to_dict() for stage in self.stages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnotDefinition':
        """Create knot definition from dictionary
        
        Args:
            data: Dictionary containing knot definition data
            
        Returns:
            KnotDefinition: Created knot definition instance
            
        Raises:
            ValueError: If data is missing required fields or has invalid values
        """
        # Validate required fields
        if 'name' not in data:
            raise ValueError("Knot definition missing required 'name' field")
        if 'stages' not in data:
            raise ValueError("Knot definition missing required 'stages' field")
        if not isinstance(data['stages'], list) or len(data['stages']) == 0:
            raise ValueError("Knot definition 'stages' must be a non-empty list")
        
        # Create stages
        stages = []
        for i, stage_data in enumerate(data['stages']):
            try:
                stages.append(KnotStage.from_dict(stage_data))
            except ValueError as e:
                raise ValueError(f"Invalid stage at index {i}: {str(e)}")
        
        # Create knot definition
        return cls(
            name=data['name'],
            description=data.get('description', ""),  # Default empty description
            stages=stages
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'KnotDefinition':
        """Load knot definition from a file (JSON format)
        
        Args:
            file_path: Path to knot definition file
            
        Returns:
            KnotDefinition: Loaded knot definition
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file has invalid format or content
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Knot definition file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in knot definition file: {str(e)}")
        
        return cls.from_dict(data)
    
    def to_file(self, file_path: Union[str, Path]):
        """Save knot definition to a file (JSON format)
        
        Args:
            file_path: Path to save knot definition file
            
        Raises:
            IOError: If file cannot be written
        """
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .knot extension if no extension provided
        if not path.suffix:
            path = path.with_suffix('.knot')
        
        # Convert to dictionary
        data = self.to_dict()
        
        # Write to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)