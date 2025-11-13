# Custom YOLO Models for Thermal Detection

This directory is for storing custom YOLO models trained specifically for thermal imaging.

## 📍 Model Storage Location

**Directory Path:**
```
pythermal/pythermal/detections/yolo/models/
```

**Full Path (after installation):**
- If installed via pip: Check your Python site-packages: `python -c "import pythermal.detections.yolo; import os; print(os.path.dirname(pythermal.detections.yolo.__file__) + '/models')"`
- If installed from source: `{project_root}/pythermal/pythermal/detections/yolo/models/`

Place your custom YOLO v11 models (`.pt` files) in this directory. The models will be automatically detected when you specify the model filename in the detector initialization.

### Supported Model Types

1. **Object Detection Models**: Custom YOLO v11 models trained for thermal object detection
2. **Pose Detection Models**: Custom YOLO v11 pose models trained for thermal pose/keypoint detection

### Usage

#### Finding the Models Directory

**Method 1: Using Python**
```python
from pathlib import Path
import pythermal.detections.yolo

models_dir = Path(pythermal.detections.yolo.__file__).parent / "models"
print(f"Models directory: {models_dir}")
```

**Method 2: Using Command Line**
```bash
python -c "from pathlib import Path; import pythermal.detections.yolo; print(Path(pythermal.detections.yolo.__file__).parent / 'models')"
```

#### Object Detection

```python
from pythermal.detections.yolo import YOLOObjectDetector

# Use custom model from this directory (relative path)
detector = YOLOObjectDetector(
    model_path="custom_thermal_object.pt",  # Relative to models/ directory
    conf_threshold=0.25,
)

# Or use absolute path
detector = YOLOObjectDetector(
    model_path="/path/to/custom_thermal_object.pt",
    conf_threshold=0.25,
)
```

#### Pose Detection

```python
from pythermal.detections.yolo import YOLOPoseDetector

# Use custom model from this directory
detector = YOLOPoseDetector(
    model_path="custom_thermal_pose.pt",  # Relative to models/ directory
    conf_threshold=0.25,
)

# Or use absolute path
detector = YOLOPoseDetector(
    model_path="/path/to/custom_thermal_pose.pt",
    conf_threshold=0.25,
)
```

### Model Training Instructions

#### TODO: Add instructions for training custom thermal models

1. **Dataset Preparation**
   - Collect thermal images with annotations
   - Format annotations in YOLO format (bounding boxes or keypoints)
   - Split into train/val/test sets

2. **Training Configuration**
   - Use YOLO v11 architecture
   - Configure for thermal image characteristics (grayscale, temperature ranges)
   - Adjust augmentation strategies for thermal data

3. **Training Process**
   - Fine-tune from pre-trained YOLO v11 models
   - Monitor validation metrics
   - Export trained model as `.pt` file

4. **Model Validation**
   - Test on thermal image sequences
   - Validate detection accuracy on thermal-specific scenarios
   - Benchmark performance on edge devices

### Model Naming Convention

Suggested naming convention:
- Object detection: `yolo11_thermal_object_{version}.pt`
- Pose detection: `yolo11_thermal_pose_{version}.pt`

Example:
- `yolo11_thermal_object_v1.pt`
- `yolo11_thermal_pose_v1.pt`

### Model Requirements

- Format: PyTorch `.pt` file
- Architecture: YOLO v11 compatible
- Input: Should accept standard image formats (BGR/RGB numpy arrays)
- Output: Should follow YOLO v11 output format

### Notes

- Default models (official YOLO v11 releases) are automatically downloaded on first use
- Custom models must be manually placed in this directory or specified with absolute paths
- Models are loaded lazily (on first detection call)
- Ensure models are compatible with your target device (CPU/GPU)

