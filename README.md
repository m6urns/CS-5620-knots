# CS-5620-knots - Knot Classification System

A real-time knot tying classification system using computer vision with OpenCV and PyTorch.

## Quick Start

1. Clone and initialize:
   ```bash
   git clone https://github.com/yourusername/knot-classification.git
   cd knot-classification
   git submodule update --init
   ```

2. Install Python dependencies:
   ```bash
   pip install -e .
   ```

3. Install UI dependencies:
   ```bash
   cd ui
   npm install
   cd ..
   ```

4. Run the backend (in one terminal):
   ```bash
   python scripts/run_square_classifier.py
   ```
   The backend API will be available at http://localhost:8000

5. Run the UI (in another terminal):
   ```bash
   cd ui
   npm run dev
   ```
   The UI will be available at http://localhost:3000

## Overview

This system provides real-time classification of knot tying stages using RGB camera input and deep learning. It features a generic knot definition system that allows you to define and classify any type of knot with arbitrary stages. The system consists of several modular components:

1. **Knot Definition**: Standardized format for defining any knot and its stages
2. **Generic Camera Interface**: Simplified camera access using OpenCV
3. **Visualization Module**: Real-time display of camera feed with overlays
4. **Data Collection Tool**: Interactive tool for collecting knot tying dataset
5. **Knot Classifier**: Machine learning model for classifying knot tying stages
6. **Training Script**: Dedicated tool for training and evaluating models
7. **Classifier API**: Web service for real-time classification

## Structure

```
knot-classifier/
│
├── README.md                          # Project documentation
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
│
├── ui/                                # UI submodule (React application)
│   ├── src/                           # Source code
│   ├── public/                        # Static assets
│   ├── package.json                   # Dependencies and scripts
│   
├── knots/                             # Core package directory  
│   ├── __init__.py                    
│   ├── knot_definition.py             # Knot structure definitions
│   ├── knot_classifier.py             # ML model for classification
│   ├── knot_data_collector.py         # Data collection utilities
│   ├── classifier_api.py              # API for inference
│   ├── utils/                         # Utility modules
│   │   ├── __init__.py
│   │   ├── generic_camera.py          # Camera utilities
│   │   └── visualization.py           # Visualization utilities
│   └── tests/                         # Unit tests
│       ├── __init__.py
│       ├── test_knot_classifier.py
│       ├── test_knot_data_collector.py
│       └── test_knot_definition.py
│
├── scripts/                           # Command-line scripts
│   ├── train_classifier.py            # Generic training script
│   ├── train_overhand_knot.py         # Knot-specific training
│   ├── run_overhand_classifier.py     # Run classifier for specific knot
│   └── collect_overhand_knot.py       # Collect data for specific knot
│
├── knot_definitions/                  # Knot definition files
│   ├── overhand_knot.knot
│   ├── figure_eight_knot.knot
│   └── square_knot.knot
│
├── data/                              # Data storage
│   └── overhand_knot/                 # Dataset for overhand knot
│
└── models/                            # Model storage
    └── overhand_knot/                 # Knot-specific models
        ├── best_model.pth             # Latest best model
        ├── confusion_matrix.png       # Evaluation results
        └── evaluation_metrics.json    # Performance metrics
```

## Requirements

## Setup

1. Clone this repository and initialize submodules:
   ```
   git clone https://github.com/yourusername/knot-classification.git
   cd knot-classification
   git submodule update --init
   ```

2. Install the package in development mode:
   ```
   pip install -e .
   ```

3. Set up the UI:
   ```
   cd ui
   npm install
   npm run dev
   ```
   The UI will be available at http://localhost:3000

4. Check camera availability:
   ```
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```
   This will print available camera indices.

## UI

The project includes a modern web interface for interacting with the knot classification system. The UI is built with React and provides:

- Real-time camera feed display
- Knot stage visualization
- Classification results
- Interactive controls

The UI runs on port 3000 by default. Make sure this port is available when starting the application.

The backend API runs on port 8000 by default. The UI will automatically connect to this port for real-time classification and camera feed.

## Knot Definitions

The system uses a standardized `.knot` file format (JSON) to define knots and their stages. Each file includes:

- Name of the knot
- Description of the knot
- An ordered list of stages, each with:
  - ID: Unique identifier (e.g., "loop")
  - Name: Human-readable name (e.g., "Loop Created")
  - Description: Detailed instructions or description

Example knot definition (overhand_knot.knot):
```json
{
  "name": "overhand_knot",
  "description": "A basic overhand knot with four stages",
  "stages": [
    {
      "id": "loose",
      "name": "Loose Rope",
      "description": "Starting position with loose rope"
    },
    {
      "id": "loop",
      "name": "Loop Created",
      "description": "A loop has been formed but not pulled through"
    },
    {
      "id": "complete",
      "name": "Knot Completed",
      "description": "Basic knot structure is complete but not tightened"
    },
    {
      "id": "tightened",
      "name": "Knot Tightened",
      "description": "The knot has been tightened and is secure"
    }
  ]
}
```

The project includes several predefined knot definitions in the `knot_definitions/` directory:
- `overhand_knot.knot`: A basic 4-stage knot
- `figure_eight_knot.knot`: A 7-stage figure-eight knot
- `square_knot.knot`: An 8-stage square (reef) knot

You can create your own custom knot definitions by following this format.

## Components

### 1. Knot Definition Module

The `KnotDefinition` class provides methods for working with knot definitions:

```python
from knots.knot_definition import KnotDefinition

# Load a knot definition
knot_def = KnotDefinition.from_file("knot_definitions/overhand_knot.knot")

# Access properties
print(f"Knot: {knot_def.name}")
print(f"Description: {knot_def.description}")
print(f"Number of stages: {knot_def.stage_count}")

# Access stages
for stage_id in knot_def.stage_ids:
    stage = knot_def.get_stage(stage_id)
    print(f"Stage: {stage.name} ({stage.id})")
    print(f"  Description: {stage.description}")
```

### 2. Generic Camera

The `GenericCamera` class provides a simple interface to access any camera compatible with OpenCV.

Example usage:
```python
from knots.utils.generic_camera import GenericCamera

# List available cameras
cameras = GenericCamera.list_available_cameras()
print("Available cameras:", cameras)

# Initialize camera (default index 0)
camera = GenericCamera(camera_index=0, resolution=(640, 480), fps=30)
camera.initialize()

# Capture frames
for _ in range(10):
    frame = camera.get_frame()
    if frame is not None:
        print(f"Captured frame: {frame.width}x{frame.height}")

# Release resources
camera.close()
```

### 3. Visualization

The `FrameVisualizer` class helps display camera frames with overlays.

Example usage:
```python
from knots.utils.visualization import FrameVisualizer, VisualizationConfig
from knots.utils.generic_camera import GenericCamera
import cv2

# Create visualization config
config = VisualizationConfig(
    window_name="Camera Feed",
    window_width=800,
    window_height=600,
    show_info=True
)

# Initialize visualizer and camera
visualizer = FrameVisualizer(config)
camera = GenericCamera(camera_index=0)
camera.initialize()

# Display frames
try:
    while True:
        frame = camera.get_frame()
        if frame is not None:
            key = visualizer.show(frame.data)
            if key == ord('q'):
                break
finally:
    camera.close()
    visualizer.close()
```

### 4. Data Collection Tool

The data collection tool lets you create a dataset for any knot defined in a `.knot` file.

Run the tool with a specific knot definition:
```
python -m scripts.collect_overhand_knot
# or
python -m scripts.collect_figure_eight_knot
# or
python -m knots.knot_data_collector --knot-def-path knot_definitions/square_knot.knot
```

Controls:
- **Space**: Capture sample
- **S**: Cycle through knot stages
- **N**: Add note to next capture
- **Q**: Quit

Dataset structure (now organized in the `data/` directory):
```
data/
└── overhand_knot_dataset/
    ├── loose/
    │   └── 20230101_120000/
    │       ├── rgb.png
    │       └── metadata.json
    ├── loop/
    │   └── ...
    ├── complete/
    │   └── ...
    ├── tightened/
    │   └── ...
    ├── overhand_knot.knot  # The knot definition used
    └── sample_counts.json
```

### 5. Training a Classifier

Train classifiers for any knot type using the simplified scripts:

```
python -m scripts.train_overhand_knot
# or
python -m scripts.train_figure_eight_knot
# or
python -m scripts.train_classifier --data-path data/square_knot_dataset --knot-def-path knot_definitions/square_knot.knot
```

Additional options:
- `--val-split 0.2`: Validation dataset portion (default: 0.2)
- `--unfreeze-epoch 10`: When to unfreeze backbone layers (default: 10)
- `--early-stopping 5`: Early stopping patience (default: 5)
- `--output-dir models`: Directory to save models (default: models)
- `--no-cuda`: Disable CUDA even if available

Output (now organized by knot type):
```
models/
└── overhand_knot/
    ├── best_model.pth             # Latest best model 
    ├── model_20250329_123045.pth  # Timestamped model version
    ├── confusion_matrix.png       # Evaluation results
    └── evaluation_metrics.json    # Performance metrics
```

### 6. Classification API

The API service provides real-time classification for any knot type.

Start the API server with a specific knot:
```
python -m scripts.run_overhand_classifier
# or
python -m scripts.run_figure_eight_classifier
# or
python -m knots.classifier_api --model-path models/custom_knot/best_model.pth --knot-def-path knot_definitions/custom_knot.knot
```

API Endpoints:
- `GET /status`: System status (including knot definition and stages)
- `GET /classification`: Latest classification result (with stage names and descriptions)
- `GET /stream`: Live video stream with classification overlay
- `GET /settings`: Current settings
- `POST /settings`: Update settings
- `POST /reload_model`: Reload the model using current settings (for hot-swapping)

#### Sequential Bias for Stage Prediction

The system now implements a sequential bias feature to improve prediction accuracy across consecutive frames:

- **Based on Natural Progression**: Applies bias based on natural progression through knot tying stages
- **Reduces Random Flickering**: Avoids rapid classification changes between unrelated stages
- **Configurable Strength**: Adjust bias intensity to match different knots and environments
- **Adjustable Decay**: Control how quickly bias diminishes for non-adjacent stages

Enable and configure sequential bias through command-line arguments or API settings:

```bash
# Enable sequential bias with custom settings
python -m scripts.run_overhand_classifier --sequential-bias --bias-strength 2.0 --bias-decay 0.3
```

Or via the API:
```python
import requests

# Enable and configure sequential bias
settings = {
    "sequential_bias": True,
    "bias_strength": 1.5,
    "bias_decay": 0.4
}
response = requests.post("http://localhost:8000/settings", json=settings)
print(response.json())
```

The sequential bias information is included in classification results:
```json
{
  "sequential_bias": {
    "enabled": true,
    "strength": 1.5,
    "decay": 0.4
  }
}
```

#### Model Hot-Swapping

The system now supports hot-swapping models without restarting the server. This allows you to:

- Change models dynamically during operation
- Test different model versions in real-time
- Update models without service interruption

There are two ways to swap models:

1. **Via the settings endpoint**:
   ```python
   import requests
   
   # Change model path
   settings = {
       "model_path": "models/overhand_knot/new_model.pth",
       "knot_def_path": "knot_definitions/overhand_knot.knot"  # Optional
   }
   response = requests.post("http://localhost:8000/settings", json=settings)
   print(response.json())
   ```

2. **Via the reload endpoint** (when you've manually replaced model files):
   ```python
   import requests
   
   # Reload current model
   response = requests.post("http://localhost:8000/reload_model")
   print(response.json())
   ```

Using curl:
```bash
# Method 1: Update settings with new model path
curl -X POST "http://localhost:8000/settings" \
     -H "Content-Type: application/json" \
     -d '{"model_path": "models/overhand_knot/new_model.pth"}'

# Method 2: Reload current model
curl -X POST "http://localhost:8000/reload_model"
```

**Notes:**
- Camera settings changes still require a server restart
- Model hot-swapping uses thread-safe mechanisms to ensure consistent operation
- If a model fails to load, the system continues using the previous model

Example API usage:
```python
import requests
import json

# Get system status
response = requests.get("http://localhost:8000/status")
print(json.dumps(response.json(), indent=2))

# Get latest classification
response = requests.get("http://localhost:8000/classification")
print(json.dumps(response.json(), indent=2))

# Update settings
new_settings = {
    "confidence_threshold": 0.7
}
response = requests.post("http://localhost:8000/settings", json=new_settings)
print(json.dumps(response.json(), indent=2))

# Hot-swap to a new model
new_settings = {
    "model_path": "models/overhand_knot/improved_model.pth"
}
response = requests.post("http://localhost:8000/settings", json=new_settings)
print(json.dumps(response.json(), indent=2))
```

## Complete Examples

### Square Knot Classification Pipeline

1. Start the backend API:
   ```bash
   python scripts/run_square_classifier.py
   ```
   This will:
   - Load the square knot model
   - Start the classification API on port 8000
   - Provide endpoints for classification and video stream

2. Start the UI:
   ```bash
   cd ui
   npm run dev
   ```
   This will:
   - Start the React development server on port 3000
   - Connect to the backend API on port 8000
   - Display the real-time classification interface

3. Access the system:
   - UI: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Video Stream: http://localhost:8000/stream
   - Classification Data: http://localhost:8000/classification

## Testing

Run the tests with pytest:
```