# YOLO v11 Detection Guide

This guide provides detailed instructions for using YOLO v11 object and pose detection with PyThermal, including how to use custom thermal-specific models.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using Default Models](#using-default-models)
4. [Using Custom Thermal Models](#using-custom-thermal-models)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Install YOLO Support

YOLO detection requires the `ultralytics` package. Install it using one of the following methods:

**Method 1: Install with pythermal extra**
```bash
pip install pythermal[yolo]
```

**Method 2: Install ultralytics separately**
```bash
pip install ultralytics>=8.0.0
```

---

## Quick Start

### Object Detection

```python
from pythermal import ThermalDevice
from pythermal.detections.yolo import YOLOObjectDetector
import cv2

device = ThermalDevice()
device.start()
shm = device.get_shared_memory()

# Initialize detector (default nano model auto-downloads on first use)
detector = YOLOObjectDetector(model_size="nano", conf_threshold=0.25)

if shm.has_new_frame():
    yuyv_frame = shm.get_yuyv_frame()
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Detect objects
    detections = detector.detect(bgr_frame)
    
    # Visualize
    vis_image = detector.visualize(bgr_frame, detections)
    cv2.imshow("Detection", vis_image)

device.stop()
```

### Pose Detection

```python
from pythermal import ThermalDevice
from pythermal.detections.yolo import YOLOPoseDetector
import cv2

device = ThermalDevice()
device.start()
shm = device.get_shared_memory()

# Initialize pose detector
pose_detector = YOLOPoseDetector(model_size="nano", conf_threshold=0.25)

if shm.has_new_frame():
    yuyv_frame = shm.get_yuyv_frame()
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Detect poses
    poses = pose_detector.detect(bgr_frame)
    
    # Visualize with skeleton and keypoints
    vis_image = pose_detector.visualize(
        bgr_frame, poses,
        show_bbox=True,
        show_keypoints=True,
        show_skeleton=True
    )
    cv2.imshow("Pose Detection", vis_image)

device.stop()
```

---

## Using Default Models

PyThermal supports official YOLO v11 models in multiple sizes:

### Available Model Sizes

| Size | Model Name | Speed | Accuracy | Use Case |
|------|------------|-------|----------|----------|
| **nano** | `yolo11n.pt` / `yolo11n-pose.pt` | Fastest | Good | Edge devices, real-time |
| **small** | `yolo11s.pt` / `yolo11s-pose.pt` | Fast | Better | Balanced performance |
| **medium** | `yolo11m.pt` / `yolo11m-pose.pt` | Medium | Good | Higher accuracy needs |
| **large** | `yolo11l.pt` / `yolo11l-pose.pt` | Slow | Better | High accuracy |
| **xlarge** | `yolo11x.pt` / `yolo11x-pose.pt` | Slowest | Best | Maximum accuracy |

### Usage

```python
# Object detection
detector = YOLOObjectDetector(
    model_size="nano",  # or "small", "medium", "large", "xlarge"
    conf_threshold=0.25,
    iou_threshold=0.45,
)

# Pose detection
pose_detector = YOLOPoseDetector(
    model_size="nano",
    conf_threshold=0.25,
    iou_threshold=0.45,
)
```

**Note:** Default models are automatically downloaded on first use. Ensure you have an internet connection for the initial download.

---

## Using Custom Thermal Models

### Model Storage Location

Custom YOLO v11 models should be placed in the models directory:

**Directory Path:**
```
pythermal/pythermal/detections/yolo/models/
```

### Finding the Models Directory

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

**Method 3: After pip installation**
```bash
MODELS_DIR=$(python -c "from pathlib import Path; import pythermal.detections.yolo; print(Path(pythermal.detections.yolo.__file__).parent / 'models')")
echo "Models directory: $MODELS_DIR"
```

### Uploading Custom Models

**If installed from source:**
```bash
cp custom_thermal_object.pt pythermal/pythermal/detections/yolo/models/
cp custom_thermal_pose.pt pythermal/pythermal/detections/yolo/models/
```

**If installed via pip:**
```bash
# Find the directory first
MODELS_DIR=$(python -c "from pathlib import Path; import pythermal.detections.yolo; print(Path(pythermal.detections.yolo.__file__).parent / 'models')")

# Copy your model
cp custom_thermal_object.pt "$MODELS_DIR/"
```

### Using Custom Models

**Option 1: Relative path (from models directory)**
```python
detector = YOLOObjectDetector(
    model_path="custom_thermal_object.pt",  # Relative to models/ directory
    conf_threshold=0.25,
)
```

**Option 2: Absolute path**
```python
detector = YOLOObjectDetector(
    model_path="/full/path/to/custom_thermal_object.pt",
    conf_threshold=0.25,
)
```

### Model Requirements

Custom models must meet the following requirements:

- **Format**: PyTorch `.pt` file
- **Architecture**: YOLO v11 compatible
- **Input**: Should accept standard image formats (BGR/RGB numpy arrays)
- **Output**: Should follow YOLO v11 output format

### Model Naming Convention

Suggested naming convention for custom models:

- Object detection: `yolo11_thermal_object_{version}.pt`
- Pose detection: `yolo11_thermal_pose_{version}.pt`

Examples:
- `yolo11_thermal_object_v1.pt`
- `yolo11_thermal_pose_v1.pt`

### Training Custom Thermal Models

**TODO: Detailed Training Instructions**

For training custom thermal models:

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

See the [Ultralytics YOLO documentation](https://docs.ultralytics.com/) for detailed training instructions.

---

## API Reference

### YOLOObjectDetector

#### Initialization

```python
YOLOObjectDetector(
    model_path: Optional[str] = None,
    model_size: str = "nano",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: Optional[str] = None,
)
```

**Parameters:**
- `model_path`: Path to custom model file. If None, uses default official model.
- `model_size`: Model size for default model ("nano", "small", "medium", "large", "xlarge").
- `conf_threshold`: Confidence threshold for detections (0.0-1.0).
- `iou_threshold`: IoU threshold for NMS (0.0-1.0).
- `device`: Device to run inference on ("cpu", "cuda", "mps", etc.). If None, auto-detects.

#### Methods

**`detect(image, classes=None, verbose=False)`**
- Detect objects in image
- Returns: List of detection dictionaries with bbox, confidence, class_id, class_name, center, width, height

**`detect_batch(images, classes=None, verbose=False)`**
- Detect objects in multiple images (batch processing)
- Returns: List of detection lists (one per image)

**`visualize(image, detections, show_labels=True, show_conf=True)`**
- Visualize detections on image
- Returns: Image with detections drawn (BGR format)

**`get_class_names()`**
- Get mapping of class IDs to class names
- Returns: Dictionary mapping class_id -> class_name

### YOLOPoseDetector

#### Initialization

```python
YOLOPoseDetector(
    model_path: Optional[str] = None,
    model_size: str = "nano",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: Optional[str] = None,
)
```

**Parameters:** Same as YOLOObjectDetector

#### Methods

**`detect(image, verbose=False)`**
- Detect poses/keypoints in image
- Returns: List of pose detection dictionaries with bbox, confidence, keypoints (17 keypoints), keypoints_dict, center, width, height

**`detect_batch(images, verbose=False)`**
- Detect poses in multiple images (batch processing)
- Returns: List of pose detection lists (one per image)

**`visualize(image, detections, show_bbox=True, show_keypoints=True, show_skeleton=True, show_labels=False, keypoint_radius=3, skeleton_thickness=2)`**
- Visualize pose detections on image
- Returns: Image with pose detections drawn (BGR format)

#### Keypoint Names

The pose detector uses 17 COCO keypoints:
1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle

---

## Examples

### Live Object Detection

See `examples/yolo_object_detection.py` for a complete example with:
- Real-time object detection on thermal camera feed
- Interactive controls (toggle labels, adjust confidence threshold)
- FPS monitoring
- Class counting

Run it:
```bash
python examples/yolo_object_detection.py
```

### Live Pose Detection

See `examples/yolo_pose_detection.py` for a complete example with:
- Real-time pose/keypoint detection
- Skeleton visualization
- Interactive controls (toggle bbox, keypoints, skeleton, labels)
- FPS monitoring

Run it:
```bash
python examples/yolo_pose_detection.py
```

---

## Troubleshooting

### ImportError: ultralytics package is required

**Problem:** `ImportError: ultralytics package is required for YOLO detection`

**Solution:**
```bash
pip install ultralytics
# or
pip install pythermal[yolo]
```

### FileNotFoundError: Model file not found

**Problem:** `FileNotFoundError: Model file not found: custom_model.pt`

**Solutions:**
1. Ensure the model file exists in the models directory
2. Check the path is correct (use absolute path if unsure)
3. Verify file permissions

**Check model location:**
```python
from pathlib import Path
import pythermal.detections.yolo

models_dir = Path(pythermal.detections.yolo.__file__).parent / "models"
print(f"Models directory: {models_dir}")
print(f"Files in directory: {list(models_dir.glob('*.pt'))}")
```

### Model download fails

**Problem:** Default models fail to download on first use

**Solutions:**
1. Check internet connection
2. Verify firewall/proxy settings
3. Manually download models from [Ultralytics releases](https://github.com/ultralytics/assets/releases)
4. Place downloaded models in the models directory

### Low detection accuracy on thermal images

**Problem:** Default models don't perform well on thermal images

**Solutions:**
1. Use custom thermal-specific models (see [Using Custom Thermal Models](#using-custom-thermal-models))
2. Adjust confidence threshold: `conf_threshold=0.15` (lower for more detections)
3. Try different model sizes (larger models may perform better)
4. Fine-tune models on thermal data

### Performance issues

**Problem:** Detection is too slow

**Solutions:**
1. Use smaller model size (`model_size="nano"` or `"small"`)
2. Use GPU if available: `device="cuda"`
3. Reduce input image resolution
4. Adjust confidence threshold to reduce post-processing

---

## Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO v11 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [PyThermal Main README](../README.md)
- [PyThermal Examples](../examples/)

