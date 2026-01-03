# 🔥 PyThermal

**A lightweight Python library for thermal sensing and analytics on Linux platforms (x86_64 and ARM).**
It provides unified APIs for recording, visualization, and intelligent analysis of thermal data from Hikvision or compatible infrared sensors.

---

## 🌡️ Features

* **Raw Frame Recording**
  Capture and store radiometric thermal frames (e.g., 96×96, 16-bit raw) with timestamps.

* **Colored Visualization**
  Generate pseudo-color thermal images (e.g., 240×240 RGB) with adjustable color maps.

* **Live Stream Interface**
  Stream frames in real time, perform temperature conversion and display dynamically.

* **Shared Memory Architecture**
  Efficient zero-copy access to thermal data via shared memory interface.

* **Multi-Device Support**
  Connect and use multiple thermal cameras simultaneously:
  
  * Consistent device ID mapping based on USB serial numbers (similar to `cv2.VideoCapture`)
  * Serial-number-based shared memory naming for strict device identification
  * Automatic device enumeration and selection
  * Device-specific shared memory segments for parallel operation
  * Persistent device mapping stored in `~/.pythermal/device_mapping.json` and `~/.pythermal/cameras` (CSV table)
  * Devices are automatically sorted by serial number for consistent ordering

* **Thermal Object Detection**
  Detect objects based on temperature ranges with clustering support:
  
  * Temperature-based object detection (default: 31-39°C for human body)
  * Adaptive human detection using environment temperature estimation
  * Object center detection and clustering
  * Temperature statistics per detected object (min / max / avg)

* **Environment & Body Temperature Estimation**
  Estimate ambient room temperature and body temperature:
  
  * Environment temperature estimation using percentile method (default: 5th percentile)
  * Body temperature estimation from environment temperature using physiological model
  * Adaptive detection thresholds based on room temperature

* **YOLO v11 Detection** (Optional)
  Advanced object and pose detection using YOLO v11:
  
  * Object detection with YOLO v11 (supports default and custom thermal models)
  * Pose/keypoint detection with 17 COCO keypoints
  * Support for custom thermal-specific models
  * Real-time inference on thermal images

* **Offline Replay and Analysis** (Future Development)
  Replay recorded sessions for algorithm benchmarking or dataset generation.

---

## 🚀 Installation

### Quick Install (Recommended)

Install from source with automatic USB setup:

```bash
git clone https://github.com/AIoT-Infrastructure/pythermal.git
cd pythermal
pip install -e .
```

**✨ Automatic USB Setup**: When you install with `pip install -e .`, the package automatically:
- Sets up USB device permissions (copies udev rules to `/etc/udev/rules.d/`)
- Adds your user to the `plugdev` group
- Reloads udev rules

You may be prompted for your password during installation to complete the USB setup. After installation:
- Disconnect and reconnect your thermal camera
- Log out and log back in (or restart) for group changes to take effect

### Alternative: Manual USB Setup

If you prefer to set up USB permissions manually, or if automatic setup didn't work:

```bash
# After installing the package
pythermal-setup-usb
```

This command will prompt for your password and set up USB permissions manually.

### Full Setup (Optional)

For a complete setup including system dependencies (FFmpeg libraries) and native compilation:

```bash
cd pythermal
./setup.sh
```

This script will:
1. Detect your system architecture (x86_64 or ARM) and install required system dependencies (FFmpeg libraries)
2. Set up USB device permissions for the thermal camera
3. Compile the native thermal recorder (`pythermal-recorder`) for your architecture

> **Note**: The native binaries are already included in the package, so compilation is only needed if you want to rebuild them.

### Install from PyPI

Install directly on a Linux device (x86_64 or ARM, e.g., x86_64 desktop, Jetson, OrangePi, Raspberry Pi):

```bash
uv pip install pythermal
```

After installation, run `pythermal-setup-usb` to set up USB permissions.

### Optional: YOLO Detection Support

To enable YOLO v11 object and pose detection, install with the `yolo` extra:

```bash
uv pip install pythermal[yolo]
```

Or install ultralytics separately:

```bash
pip install ultralytics>=8.0.0
```

> ✅ **Bundled Native Runtime**
> The package ships with the native thermal recorder (`pythermal-recorder`) and required shared libraries (`.so` files) for both x86_64 (`pythermal/_native/linux64/`) and ARM (`pythermal/_native/armLinux/`). The library automatically detects your system architecture and uses the appropriate binaries.

---

## 🧠 Quick Start

### 1. Unified Capture Interface

The `ThermalCapture` class provides a unified interface for both live camera feeds and recorded sequences. It's similar to `cv2.VideoCapture` and automatically handles device initialization.

```python
from pythermal import ThermalCapture

# For live camera (default - uses smallest available device)
capture = ThermalCapture()  # or ThermalCapture(0) or ThermalCapture(None)

# For specific device (by consistent device ID)
capture = ThermalCapture(device_index=1)  # Use device with ID 1

# For recorded sequence
capture = ThermalCapture("recordings/thermal_20240101.tseq")

# Check for new frame
if capture.has_new_frame():
    # Get metadata
    metadata = capture.get_metadata()
    print(f"Frame {metadata.seq}: {metadata.min_temp:.1f}°C - {metadata.max_temp:.1f}°C")
    
    # Get YUYV frame (240x240)
    yuyv_frame = capture.get_yuyv_frame()
    
    # Get temperature array (96x96, uint16)
    temp_array = capture.get_temperature_array()
    
    # Mark frame as read
    capture.mark_frame_read()

# Cleanup
capture.release()
```

Or use as a context manager:

```python
with ThermalCapture() as capture:
    if capture.has_new_frame():
        metadata = capture.get_metadata()
        yuyv_frame = capture.get_yuyv_frame()
        temp_array = capture.get_temperature_array()
        capture.mark_frame_read()
    # Automatically releases on exit
```

---

### 2. Live View

Display real-time thermal imaging feed:

```python
from pythermal import ThermalLiveView

# For live camera (default)
viewer = ThermalLiveView()
viewer.run()  # Opens OpenCV window with live thermal feed

# For recorded sequence
viewer = ThermalLiveView("recordings/thermal_20240101.tseq")
viewer.run()  # Replays recorded sequence
```

**Controls:**
- Press `q` to quit
- Press `t` to toggle between YUYV view and temperature view
- Press `+`/`-` to adjust contrast (CLAHE enhancement)
- Move mouse over image to see temperature at cursor position
- Press `SPACE` to pause/resume (for recorded sequences)

The live view displays room temperature estimation in the overlay, calculated using the 5th percentile method with smoothing.

---

### 3. Record Thermal Frames

```python
from pythermal import ThermalRecorder
import time

rec = ThermalRecorder(output_dir="recordings", color=True)
rec.start()              # Starts device and begins recording
rec.record_loop(duration=10)  # Record for 10 seconds
rec.stop()               # Stop recording
```

This records both:

* Raw temperature frames (`96×96`, uint16)
* YUYV visual frames (`240×240`)
* Colored RGB frames (`240×240`, uint8 RGB) if `color=True`

---

### 4. Access Thermal Data Directly

Using the unified `ThermalCapture` interface (recommended):

```python
from pythermal import ThermalCapture

capture = ThermalCapture()  # Live camera, or pass file path for recorded data

# Check for new frame
if capture.has_new_frame():
    # Get metadata
    metadata = capture.get_metadata()
    print(f"Frame {metadata.seq}: {metadata.min_temp:.1f}°C - {metadata.max_temp:.1f}°C")
    
    # Get YUYV frame (240x240)
    yuyv_frame = capture.get_yuyv_frame()
    
    # Get temperature array (96x96, uint16)
    temp_array = capture.get_temperature_array()
    
    # Mark frame as read
    capture.mark_frame_read()

capture.release()
```

**Multi-Device Usage**

When multiple thermal cameras are connected, PyThermal automatically assigns consistent device IDs based on USB serial numbers. This ensures that the same physical device always gets the same ID, even after reconnection:

```python
from pythermal import ThermalCapture

# Use device with ID 0 (first device, or smallest available)
capture0 = ThermalCapture(device_index=0)

# Use device with ID 1 (second device)
capture1 = ThermalCapture(device_index=1)

# If no device_index is specified, uses smallest available device ID
capture_auto = ThermalCapture()  # Automatically selects smallest available device

# Each device uses separate shared memory (named by serial number):
# Device 0 (serial EA4782688): /dev/shm/yuyv240_shm_EA4782688
# Device 1 (serial EA4782767): /dev/shm/yuyv240_shm_EA4782767
# Device 2 (serial EA4782845): /dev/shm/yuyv240_shm_EA4782845
# etc.
```

**Device Mapping**

Device IDs are stored persistently in two formats:
1. **JSON mapping** (`~/.pythermal/device_mapping.json`): Maps USB serial numbers to consistent device IDs
2. **CSV table** (`~/.pythermal/cameras`): Human-readable table format for easy viewing/editing

This ensures:
- Same device always gets the same ID (even after reboot)
- Device IDs remain stable across sessions
- Devices are automatically sorted by serial number for consistent ordering
- Automatic selection of smallest available device when `device_index=None`
- Strict device identification using serial numbers in shared memory names

**Advanced: Direct Shared Memory Access**

For advanced use cases, you can access the shared memory interface directly:

```python
from pythermal import ThermalDevice, ThermalSharedMemory

# Use specific device (by consistent device ID)
device = ThermalDevice(device_index=1)
device.start()
shm = device.get_shared_memory()

if shm.has_new_frame():
    metadata = shm.get_metadata()
    yuyv_frame = shm.get_yuyv_frame()
    temp_array = shm.get_temperature_array()
    
    # Get temperature map in Celsius (96x96, float32)
    temp_celsius = shm.get_temperature_map_celsius()
    
    shm.mark_frame_read()

device.stop()
```

---

### 5. Detect Objects in Thermal Images

Detect objects based on temperature ranges and visualize them with clustering:

```python
from pythermal import ThermalCapture, detect_object_centers, cluster_objects

capture = ThermalCapture()  # Live camera, or pass file path for recorded data

if capture.has_new_frame():
    metadata = capture.get_metadata()
    temp_array = capture.get_temperature_array()
    
    # Detect objects in temperature range (default: 31-39°C for human body)
    objects = detect_object_centers(
        temp_array=temp_array,
        min_temp=metadata.min_temp,
        max_temp=metadata.max_temp,
        temp_min=31.0,  # Minimum temperature threshold
        temp_max=39.0,  # Maximum temperature threshold
        min_area=50     # Minimum area in pixels
    )
    
    # Cluster nearby objects together
    clusters = cluster_objects(objects, max_distance=30.0)
    
    # Each object contains: center_x, center_y, width, height, area,
    # avg_temperature, max_temperature, min_temperature
    for obj in objects:
        print(f"Object at ({obj.center_x:.1f}, {obj.center_y:.1f}): "
              f"{obj.avg_temperature:.1f}°C")
    
    capture.mark_frame_read()

capture.release()
```

See `examples/detect_objects.py` for a complete visualization example. All examples support both live camera feeds and recorded sequences using the `ThermalCapture` interface.

#### Adaptive Human Detection

Detect humans using adaptive temperature thresholds based on environment temperature:

```python
from pythermal import ThermalCapture, detect_humans_adaptive

capture = ThermalCapture()  # Live camera, or pass file path for recorded data

if capture.has_new_frame():
    metadata = capture.get_metadata()
    temp_array = capture.get_temperature_array()
    
    # Adaptive detection (auto-estimates room temperature)
    objects = detect_humans_adaptive(
        temp_array=temp_array,
        min_temp=metadata.min_temp,
        max_temp=metadata.max_temp
    )
    
    # Or specify room temperature
    objects = detect_humans_adaptive(
        temp_array=temp_array,
        min_temp=metadata.min_temp,
        max_temp=metadata.max_temp,
        environment_temp=22.0  # Known room temperature
    )
    
    capture.mark_frame_read()

capture.release()
```

The adaptive detection uses the formula `Ts = Te + α × (Tc − Te)` where:
- `Ts` = Skin temperature (estimated body temperature)
- `Te` = Environment temperature
- `Tc` = Core body temperature (37°C)
- `α` = Blood flow regulation coefficient (0.4-0.7 for face/torso)

---

### 6. YOLO v11 Object and Pose Detection

Detect objects and human poses using YOLO v11 models:

```python
from pythermal import ThermalCapture
from pythermal.detections.yolo import YOLOObjectDetector, YOLOPoseDetector
import cv2

capture = ThermalCapture()  # Live camera, or pass file path for recorded data

# Initialize YOLO detectors (default models auto-download on first use)
object_detector = YOLOObjectDetector(model_size="nano")  # Options: nano, small, medium, large, xlarge
pose_detector = YOLOPoseDetector(model_size="nano")

if capture.has_new_frame():
    yuyv_frame = capture.get_yuyv_frame()
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Object detection
    objects = object_detector.detect(bgr_frame)
    for obj in objects:
        print(f"Detected {obj['class_name']} with confidence {obj['confidence']:.2f}")
    
    # Pose detection
    poses = pose_detector.detect(bgr_frame)
    for pose in poses:
        print(f"Detected person with {len(pose['keypoints'])} keypoints")
    
    # Visualize
    vis_image = object_detector.visualize(bgr_frame, objects)
    # or
    vis_image = pose_detector.visualize(bgr_frame, poses)
    
    capture.mark_frame_read()

capture.release()
```

#### Using Custom Thermal Models

Place your custom YOLO v11 models (`.pt` files) in:
```
pythermal/pythermal/detections/yolo/models/
```

**Quick usage:**
```python
# Use custom model from models directory
detector = YOLOObjectDetector(model_path="custom_thermal_object.pt")
```

📖 **For detailed instructions** on finding the models directory, uploading custom models, training, and troubleshooting, see the [YOLO Detection Guide](docs/YOLO_DETECTION.md).

See `examples/yolo_object_detection.py` and `examples/yolo_pose_detection.py` for complete examples.

---

## 🔌 Multi-Device Support

PyThermal supports connecting and using multiple thermal cameras simultaneously. Each device is assigned a consistent device ID based on its USB serial number, ensuring stable identification across sessions and reboots.

### How It Works

1. **Device Enumeration**: When you connect a thermal camera, PyThermal queries the USB SDK to get the device's serial number.

2. **Consistent ID Mapping**: Each unique serial number is mapped to a consistent device ID (0, 1, 2, ...) stored in `~/.pythermal/device_mapping.json`.

3. **Automatic Selection**: If no `device_index` is specified, PyThermal automatically uses the smallest available device ID.

4. **Device-Specific Shared Memory**: Each device uses its own shared memory segment named by serial number:
   - Device 0 (serial EA4782688): `/dev/shm/yuyv240_shm_EA4782688`
   - Device 1 (serial EA4782767): `/dev/shm/yuyv240_shm_EA4782767`
   - Device 2 (serial EA4782845): `/dev/shm/yuyv240_shm_EA4782845`
   - etc.
   
   This serial-number-based naming ensures strict device identification - each physical camera always uses the same shared memory name, regardless of USB port or enumeration order.

### Usage Examples

**Basic Usage:**
```python
from pythermal import ThermalCapture

# Automatically use smallest available device
capture = ThermalCapture()

# Use specific device by ID
capture0 = ThermalCapture(device_index=0)
capture1 = ThermalCapture(device_index=1)
```

**Parallel Operation:**
```python
from pythermal import ThermalCapture
import threading

def capture_from_device(device_id):
    capture = ThermalCapture(device_index=device_id)
    while True:
        if capture.has_new_frame():
            metadata = capture.get_metadata()
            print(f"Device {device_id}: {metadata.avg_temp:.1f}°C")
            capture.mark_frame_read()

# Capture from multiple devices in parallel
thread0 = threading.Thread(target=capture_from_device, args=(0,))
thread1 = threading.Thread(target=capture_from_device, args=(1,))
thread0.start()
thread1.start()
```

**Command-Line Usage:**
```bash
# Use device 0 (default)
python examples/live_view.py

# Use device 1
python examples/live_view.py --device-index 1

# Record from device 0
python examples/record_thermal.py --duration 10

# Record from device 1
python examples/record_thermal.py --duration 10 --device-index 1
```

### Device Mapping Files

The device mapping is stored in two formats:

**JSON Format** (`~/.pythermal/device_mapping.json`):
```json
{
  "EA4782688": 0,
  "EA4782767": 1,
  "EA4782845": 2
}
```

**CSV Table** (`~/.pythermal/cameras`):
```csv
serial_number,device_index
EA4782688,0
EA4782767,1
EA4782845,2
```

This ensures:
- Same physical device always gets the same ID
- Device IDs persist across reboots
- Devices are automatically sorted by serial number for consistent ordering
- Easy identification of devices by serial number
- Human-readable CSV format for easy viewing/editing
- Strict device identification using serial numbers in shared memory names

### Troubleshooting Multi-Device Issues

* **Device ID changes after reconnection**
  - Check that USB serial numbers are being read correctly
  - Verify `~/.pythermal/device_mapping.json` exists and is writable
  - Try removing the mapping file and letting PyThermal recreate it

* **Cannot access multiple devices simultaneously**
  - Ensure each device uses a different `device_index`
  - Check that shared memory segments are unique for each device
  - Verify USB permissions are set up correctly for all devices

* **Wrong device selected**
  - Explicitly specify `device_index` instead of relying on automatic selection
  - Check the device mapping file to see which serial number maps to which ID
  - Use `enumerate_devices` helper to list all connected devices

---

## 🧩 Command Line Interface

| Command                | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `pythermal-preview` | Live preview with temperature overlay           |
| `pythermal-setup-usb` | Set up USB device permissions for thermal camera |

Examples:

```bash
# Live preview
pythermal-preview

# Set up USB permissions (if not done during install)
pythermal-setup-usb
```

The `pythermal-setup-usb` command will:
- Copy udev rules to `/etc/udev/rules.d/`
- Add your user to the `plugdev` group
- Reload udev rules

After running, disconnect and reconnect your thermal camera, and log out/in for changes to take effect.

---

## 🧰 API Overview

### Core Classes

| Class                 | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `ThermalCapture`      | **Unified interface** for live camera feeds and recorded sequences (recommended) |
| `ThermalDevice`       | Manages thermal camera initialization via subprocess and shared memory access (advanced) |
| `ThermalSharedMemory` | Reads thermal data from shared memory (YUYV frames, temperature arrays, metadata) (advanced) |
| `ThermalSequenceReader` | Reads pre-recorded thermal sequences from `.tseq` files |
| `ThermalRecorder`     | Records raw and colored frames to files        |
| `ThermalLiveView`     | Displays live thermal imaging feed with OpenCV  |
| `FrameMetadata`       | Named tuple containing frame metadata (seq, flag, dimensions, temperatures) |

### Detection Classes

| Class                 | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `DetectedObject`      | Dataclass representing a detected object with center, size, and temperature stats |
| `BackgroundSubtractor` | Background subtraction for motion detection using running average |
| `ROI`                 | Region of Interest definition with optional temperature thresholds |
| `ROIManager`          | Manages multiple ROIs for zone monitoring and filtering |
| `YOLOObjectDetector`  | YOLO v11 object detector (requires `ultralytics` package) |
| `YOLOPoseDetector`    | YOLO v11 pose/keypoint detector (requires `ultralytics` package) |

### Detection Functions

| Function              | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `detect_object_centers` | Detect object centers from temperature map based on temperature range |
| `detect_humans_adaptive` | Adaptive human detection using environment temperature estimation |
| `detect_moving_objects` | Detect moving objects using background subtraction |
| `cluster_objects`    | Cluster detected objects that are close to each other |

### Utility Functions

| Function              | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `estimate_environment_temperature_v1` | Estimate room temperature using 5th percentile method (with smoothing) |
| `estimate_body_temperature` | Estimate body (skin) temperature from environment temperature |
| `estimate_body_temperature_range` | Get temperature range for different body parts (hands/feet vs face/torso) |

### Shape Analysis Functions

| Function              | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `calculate_aspect_ratio` | Calculate aspect ratio (width/height) of detected object |
| `calculate_compactness` | Calculate compactness (circularity approximation) |
| `calculate_circularity` | Calculate true circularity from contour |
| `calculate_convexity_ratio` | Calculate convexity ratio from contour |
| `filter_by_aspect_ratio` | Filter objects by aspect ratio |
| `filter_by_compactness` | Filter objects by compactness |
| `filter_by_area`      | Filter objects by area (min/max) |
| `filter_by_shape`     | Filter objects by multiple shape criteria |

### YOLO Detection Methods

| Method                | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `YOLOObjectDetector.detect()` | Detect objects in image, returns list of detections with bbox, class, confidence |
| `YOLOObjectDetector.visualize()` | Draw bounding boxes and labels on image |
| `YOLOPoseDetector.detect()` | Detect poses/keypoints in image, returns list of poses with 17 keypoints |
| `YOLOPoseDetector.visualize()` | Draw skeleton, keypoints, and bounding boxes on image |

---

## 🧪 Requirements

* Python ≥ 3.9
* Linux environment (x86_64 or ARM, e.g., x86_64 desktop, Jetson, OrangePi, Raspberry Pi)
* NumPy, OpenCV (auto-installed via pip)
* Thermal camera connected via USB
* Proper USB permissions (automatically set up during `pip install -e .`, or manually via `pythermal-setup-usb`)

### Optional Dependencies

* **ultralytics ≥ 8.0.0**: Required for YOLO v11 detection features
  ```bash
  pip install ultralytics
  # or
  pip install pythermal[yolo]
  ```

---

## ⚙️ Architecture

### Native Runtime

The library uses a native binary (`pythermal-recorder`) that runs as a separate process and writes thermal data to shared memory (`/dev/shm/yuyv240_shm`). The Python library communicates with this process via shared memory for efficient zero-copy data access.

### Bundled Files

The package includes native files for both x86_64 and ARM architectures:

**x86_64 (`pythermal/_native/linux64/`):**
```
pythermal/_native/linux64/
├── pythermal-recorder      # Main thermal recorder executable
├── libHCUSBSDK.so            # Hikvision USB SDK library
├── libhpr.so                 # Hikvision processing library
├── libusb-1.0.so*            # USB library dependencies
└── libuvc.so                  # UVC library
```

**ARM (`pythermal/_native/armLinux/`):**
```
pythermal/_native/armLinux/
├── pythermal-recorder      # Main thermal recorder executable
├── libHCUSBSDK.so            # Hikvision USB SDK library
├── libhpr.so                 # Hikvision processing library
├── libusb-1.0.so*            # USB library dependencies
└── libuvc.so                  # UVC library
```

The library automatically detects your system architecture and loads the appropriate binaries.

### Shared Memory Layout

Each device uses its own shared memory segment named by serial number:
- Device 0 (serial EA4782688): `/dev/shm/yuyv240_shm_EA4782688`
- Device 1 (serial EA4782767): `/dev/shm/yuyv240_shm_EA4782767`
- Device 2 (serial EA4782845): `/dev/shm/yuyv240_shm_EA4782845`

**Serial-Number-Based Naming**: PyThermal uses USB serial numbers in shared memory names to ensure strict device identification. This means:
- Each physical camera always uses the same shared memory name
- No confusion even if devices are reconnected in different USB ports
- Consistent device mapping regardless of enumeration order
- etc.

The shared memory (`/dev/shm/yuyv240_shm` or `/dev/shm/yuyv240_shm_{device_id}`) contains:

```
Offset          Size            Content
0               FRAME_SZ        YUYV frame data (240×240×2 bytes)
FRAME_SZ        TEMP_DATA_SIZE  Temperature array (96×96×2 bytes, uint16)
FRAME_SZ+TEMP   ...             Metadata:
                                - seq (4 bytes, uint32)
                                - flag (4 bytes, uint32, 1=new, 0=consumed)
                                - width (4 bytes, uint32)
                                - height (4 bytes, uint32)
                                - min_temp (4 bytes, float)
                                - max_temp (4 bytes, float)
                                - avg_temp (4 bytes, float)
                                - reserved (4 bytes)
```

### Process Management

The `ThermalDevice` class:
1. Starts `pythermal-recorder` as a subprocess
2. Waits for shared memory to become available
3. Provides access to thermal data via `ThermalSharedMemory`
4. Automatically cleans up the process on exit

### Troubleshooting

* **`FileNotFoundError: pythermal-recorder not found`**
  Make sure you've run `setup.sh` to compile the native binaries, and that the package was installed correctly.

* **`PermissionError: pythermal-recorder is not executable`**
  Run `chmod +x` on the executable, or reinstall the package.

* **`TimeoutError: Shared memory did not become available`**
  - Check that the thermal camera is connected via USB
  - Verify USB permissions are set up correctly (run `pythermal-setup-usb` or `setup.sh`)
  - Try disconnecting and reconnecting the camera
  - Log out and log back in (or restart) after setting up USB permissions
  - Check that no other process is using the thermal camera

* **`RuntimeError: Thermal recorder process exited unexpectedly`**
  Check the process output for error messages. Common issues:
  - Camera not detected
  - Missing USB permissions (run `pythermal-setup-usb` to fix)
  - Missing shared libraries (check `LD_LIBRARY_PATH`)
  
* **USB Permission Issues**
  - If you get permission errors accessing the thermal camera:
    1. Run `pythermal-setup-usb` (or `sudo pythermal-setup-usb`)
    2. Disconnect and reconnect your thermal camera
    3. Log out and log back in (or restart your system)
    4. Verify with `lsusb` that your camera is detected

* **`ImportError: ultralytics package is required for YOLO detection`**
  Install the ultralytics package:
  ```bash
  pip install ultralytics
  # or
  pip install pythermal[yolo]
  ```

* **`FileNotFoundError: Model file not found`**
  - For custom models, ensure the `.pt` file is in `pythermal/pythermal/detections/yolo/models/`
  - Or provide the absolute path to the model file
  - Default models are automatically downloaded on first use (check internet connection)
  - See [YOLO Detection Guide](docs/YOLO_DETECTION.md#troubleshooting) for detailed troubleshooting

---

## 📦 Directory Structure

```
pythermal/
├── pythermal/
│   ├── __init__.py
│   ├── core/                   # Core thermal camera components
│   │   ├── __init__.py
│   │   ├── device.py              # ThermalDevice class (manages subprocess)
│   │   ├── thermal_shared_memory.py  # Shared memory reader
│   │   ├── sequence_reader.py    # ThermalSequenceReader for recorded files
│   │   └── capture.py            # ThermalCapture unified interface
│   ├── record.py              # ThermalRecorder class
│   ├── live_view.py           # ThermalLiveView class
│   ├── detections/            # Object detection module
│   │   ├── __init__.py
│   │   ├── utils.py           # Shared utilities and shape analysis
│   │   ├── temperature_detection.py  # Temperature-based detection
│   │   ├── motion_detection.py       # Background subtraction and motion detection
│   │   ├── roi.py             # ROI management and zone monitoring
│   │   └── yolo/              # YOLO v11 detection module
│   │       ├── __init__.py
│   │       ├── object_detection.py   # YOLO object detection
│   │       ├── pose_detection.py    # YOLO pose detection
│   │       └── models/              # Custom thermal models directory
│   │           └── README.md         # Instructions for custom models
│   ├── usb_setup/             # USB setup scripts (included in package)
│   │   ├── setup.sh           # USB permissions setup script
│   │   ├── setup-thermal-permissions.sh
│   │   └── 99-thermal-camera.rules  # udev rules file
│   └── _native/
│       ├── linux64/           # x86_64 binaries
│       │   ├── pythermal-recorder
│       │   └── *.so            # Native libraries
│       └── armLinux/          # ARM binaries
│           ├── pythermal-recorder
│           └── *.so            # Native libraries
├── examples/                  # Example scripts
│   ├── live_view.py
│   ├── record_thermal.py
│   ├── detect_objects.py      # Object detection visualization example
│   ├── detect_motion.py       # Motion detection example
│   ├── detect_roi.py          # ROI zone monitoring example
│   ├── yolo_object_detection.py  # YOLO object detection example
│   └── yolo_pose_detection.py   # YOLO pose detection example
├── setup.sh                   # Full setup script (permissions, dependencies, compilation)
├── setup.py                   # Python package setup (includes automatic USB setup)
└── README.md
```

---

## 📚 References

```
@inproceedings{zeng2025thermikit,
  title={ThermiKit: Edge-Optimized LWIR Analytics with Agent-Driven Interactions},
  author={Zeng, Lan and Huang, Chunhao and Xie, Ruihan and Huang, Zhuohan and Guo, Yunqi and He, Lixing and Xie, Zhiyuan and Xing, Guoliang},
  booktitle={Proceedings of the 2025 ACM International Workshop on Thermal Sensing and Computing},
  pages={40--46},
  year={2025}
}
```

---

## 📄 License

This library is released under the **Apache 2.0 License** for research and non-commercial use.
Only the compiled native library (`.so`) is shipped; no vendor source or headers are distributed.

---

## 💡 Acknowledgements

**🏫 Developed by AIoT Lab, CUHK**  
