# Synthetic Thermal Data Generation

This module provides functionality to generate synthetic thermal frames (`.tframe` format) from RGB images. It uses YOLO for human detection and pose estimation to create realistic thermal representations.

## Features

- **Image Preprocessing**: Automatic resize/crop to 240x240 pixels
- **Human Segmentation**: Uses YOLO object detection to identify persons
- **Pose Estimation**: YOLO pose detection to identify body parts
- **Temperature Mapping**: Assigns different temperatures to:
  - Body parts (head, torso, limbs) - temperatures estimated from ambient temp using physiological model
  - Exposed skin - detected using RGB color analysis, warmer than clothing
  - Clothing - cooler regions
  - Background - ambient temperature
- **Frame Generation**: Creates `.tframe` files compatible with PyThermal

## Installation

The synthesis module requires YOLO dependencies:

```bash
pip install pythermal[yolo]
# or
pip install ultralytics>=8.0.0
```

## Quick Start

### Basic Usage

```python
from pythermal.synthesis import SyntheticThermalGenerator

# Initialize generator
generator = SyntheticThermalGenerator(
    core_temp=37.0,      # Core body temperature (°C)
    clothing_temp=28.0,  # Clothing temperature (°C)
    ambient_temp=22.0,   # Ambient/background temperature (°C)
    # Body part temperatures are automatically estimated from ambient_temp
    # using estimate_body_temperature() with different alpha values:
    # - Head: alpha=0.65 (warmer)
    # - Torso: alpha=0.6
    # - Limbs: alpha=0.4
    # - Extremities: alpha=0.25 (cooler)
)

# Generate thermal frame from RGB image
thermal_frame, rendered_image = generator.generate_from_image(
    "input_image.jpg",
    output_path="output.tframe",
    view_mode="temperature"
)
```

### From NumPy Array

```python
import cv2
import numpy as np
from pythermal.synthesis import SyntheticThermalGenerator

# Load RGB image
rgb_image = cv2.imread("input.jpg")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Generate thermal frame
generator = SyntheticThermalGenerator()
thermal_frame, rendered_image = generator.generate_from_array(
    rgb_image,
    output_path="output.tframe"
)
```

## Configuration

### Temperature Parameters

- `core_temp`: Core body temperature (default: 37.0°C)
- `clothing_temp`: Clothing temperature (default: 28.0°C)
- `ambient_temp`: Background/ambient temperature (default: 22.0°C)
  - Body part temperatures are automatically estimated from `ambient_temp` using `estimate_body_temperature()`
  - Uses physiological model: Ts = Te + α × (Tc − Te)
  - Different alpha values for different body parts:
    - Head/face: α=0.65 (warmer, exposed skin)
    - Torso: α=0.6
    - Limbs: α=0.4
    - Hands/feet: α=0.25 (cooler, extremities)

### Model Configuration

- `model_size`: YOLO object detection model size ("nano", "small", "medium", "large", "xlarge")
- `pose_model_size`: YOLO pose detection model size
- `conf_threshold`: Confidence threshold for detections (default: 0.25)
- `device`: Device to run inference on ("cpu", "cuda", etc.)

## Example Script

See `examples/synthetic_thermal.py` for a complete example:

```bash
python examples/synthetic_thermal.py input_image.jpg -o output.tframe
```

## Architecture

The module is designed to be extendable for video synthesis:

- **ImageProcessor**: Handles image preprocessing (resize/crop)
- **HumanSegmenter**: Segments humans using YOLO object detection
- **TemperatureMapper**: Maps temperatures based on pose keypoints
- **ThermalFrameGenerator**: Generates ThermalFrame objects
- **SyntheticThermalGenerator**: Main API coordinating all components

## Future: Video Synthesis

The module structure is designed to support video synthesis. Future enhancements may include:

- Batch processing for video frames
- Temporal consistency for temperature mapping
- Motion-aware temperature assignment
- Video sequence export to `.tseq` format

## Notes

- Input images are automatically resized/cropped to 240x240 pixels
- Temperature mapping uses pose keypoints to identify body regions
- Multiple people in the image are handled automatically
- The generated `.tframe` files are compatible with PyThermal's existing tools

