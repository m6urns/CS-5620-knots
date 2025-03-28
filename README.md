

# CS-5620-knots - Knot Classification System

A real-time knot tying classification system using computer vision with OpenCV and PyTorch.

## Overview

This system provides real-time classification of overhand knot tying stages using RGB camera input and deep learning. It consists of several modular components that can be used independently or together:

1. **Generic Camera Interface**: Simplified camera access using OpenCV
2. **Visualization Module**: Real-time display of camera feed with overlays
3. **Data Collection Tool**: Interactive tool for collecting knot tying dataset
4. **Knot Classifier**: Machine learning model for classifying knot tying stages
5. **Training Script**: Dedicated tool for training and evaluating models
6. **Classifier API**: Web service for real-time classification

## Requirements

### Hardware
- Any webcam or camera compatible with OpenCV
- Computer with CPU/GPU capable of running PyTorch

### Software Dependencies
```
numpy>=1.20.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
fastapi>=0.68.0
uvicorn>=0.15.0
matplotlib>=3.4.0  # For training visualization only
seaborn>=0.11.0    # For training visualization only
scikit-learn>=0.24.0  # For evaluation metrics
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/knot-classification.git
   cd knot-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Check camera availability:
   ```
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```
   This will print available camera indices.

## Components

### 1. Generic Camera

The `GenericCamera` class provides a simple interface to access any camera compatible with OpenCV.

Example usage:
```python
from generic_camera import GenericCamera

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

You can also use context manager syntax:
```python
with GenericCamera(camera_index=0) as camera:
    frame = camera.get_frame()
    # Process frame...
```

### 2. Visualization

The `FrameVisualizer` class helps display camera frames with overlays.

Example usage:
```python
from visualization import FrameVisualizer, VisualizationConfig
from generic_camera import GenericCamera
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

### 3. Data Collection Tool

The data collection tool helps create a dataset of knot tying stages.

Run the tool:
```
python knot_data_collector.py --camera-index 0 --width 640 --height 480
```

Controls:
- **Space**: Capture sample
- **S**: Cycle through knot stages (loose, loop, complete, tightened)
- **N**: Add note to next capture
- **Q**: Quit

Dataset structure:
```
overhand_knot_dataset/
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
└── sample_counts.json
```

### 4. Training a Classifier

The training script handles dataset loading, model training, and evaluation.

Run the training:
```
python train_classifier.py --data-path overhand_knot_dataset --epochs 30 --batch-size 4
```

Additional options:
- `--val-split 0.2`: Validation dataset portion (default: 0.2)
- `--unfreeze-epoch 10`: When to unfreeze backbone layers (default: 10)
- `--early-stopping 5`: Early stopping patience (default: 5)
- `--output-dir models`: Directory to save models (default: models)
- `--no-cuda`: Disable CUDA even if available

Output:
- `models/best_model_TIMESTAMP.pth`: Best model weights
- `models/latest_model.pth`: Latest model weights
- `models/confusion_matrix.png`: Confusion matrix visualization

### 5. Classification API

The API service provides real-time classification through a web interface.

Start the API server:
```
python classifier_api.py --camera-index 0 --model-path models/best_model.pth
```

API Endpoints:
- `GET /status`: System status
- `GET /classification`: Latest classification result
- `GET /stream`: Live video stream with classification overlay
- `GET /settings`: Current settings
- `POST /settings`: Update settings

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
```

Access the video stream in a browser:
```
http://localhost:8000/stream
```

### 6. Demo Application

A simple demo application for testing the camera and visualization:

```
python demo_camera.py --camera-index 0 --width 640 --height 480
```

## Examples

### Complete Classification Pipeline

1. Collect data:
   ```
   python knot_data_collector.py --camera-index 0
   ```

2. Train model:
   ```
   python train_classifier.py --data-path overhand_knot_dataset
   ```

3. Start classification API:
   ```
   python classifier_api.py --camera-index 0 --model-path models/best_model.pth
   ```

4. View the results in a browser:
   ```
   http://localhost:8000/stream
   ```