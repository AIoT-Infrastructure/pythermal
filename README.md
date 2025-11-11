# 🔥 PyThermal

**A lightweight Python library for thermal sensing and analytics on ARM Linux platforms.**
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

* **Thermal Analytics** (Future Development)
  Built-in lightweight models for:

  * Human body detection
  * Skeleton (pose) detection
  * ROI temperature statistics (max / min / avg)

* **Offline Replay and Analysis** (Future Development)
  Replay recorded sessions for algorithm benchmarking or dataset generation.

---

## 🚀 Installation

### Prerequisites

Before installing the Python package, you need to set up the thermal camera permissions and native runtime:

```bash
cd pythermal-devkit
./setup.sh
```

This script will:
1. Install required system dependencies (cross-compiler, FFmpeg libraries)
2. Set up USB device permissions for the thermal camera
3. Compile the native thermal recorder (`pythermal-recorder`)

After running `setup.sh`, you may need to:
- Disconnect and reconnect your thermal camera
- Log out and log back in (or restart) for permissions to take effect

### Install Python Package

Install directly on an ARM Linux device (e.g., Jetson, OrangePi, Raspberry Pi):

```bash
uv pip install pythermal
```

Or from source:

```bash
git clone https://github.com/<your-org>/pythermal.git
cd pythermal-devkit
uv pip install .
```

> ✅ **Bundled Native Runtime**
> The package ships with the native thermal recorder (`pythermal-recorder`) and required shared libraries (`.so` files) under `pythermal/_native/armLinux/`.

---

## 🧠 Quick Start

### 1. Initialize Thermal Device

The `ThermalDevice` class manages the thermal camera initialization by starting `pythermal-recorder` in a separate process and providing access to thermal data via shared memory.

```python
from pythermal import ThermalDevice

# Create and start thermal device
device = ThermalDevice()
device.start()  # Starts pythermal-recorder subprocess and initializes shared memory

# Access shared memory for reading thermal data
shm = device.get_shared_memory()

# When done, stop the device
device.stop()
```

Or use as a context manager:

```python
with ThermalDevice() as device:
    shm = device.get_shared_memory()
    # Use shared memory...
    # Device automatically stops on exit
```

---

### 2. Live View

Display real-time thermal imaging feed:

```python
from pythermal import ThermalLiveView

viewer = ThermalLiveView()
viewer.run()  # Opens OpenCV window with live thermal feed
```

Or with a shared device:

```python
from pythermal import ThermalDevice, ThermalLiveView

device = ThermalDevice()
device.start()

viewer = ThermalLiveView(device=device)
viewer.run()  # Uses the shared device

device.stop()
```

**Controls:**
- Press `q` to quit
- Press `t` to toggle between YUYV view and temperature view
- Move mouse over image to see temperature at cursor position

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

```python
from pythermal import ThermalDevice, ThermalSharedMemory

device = ThermalDevice()
device.start()

shm = device.get_shared_memory()

# Check for new frame
if shm.has_new_frame():
    # Get metadata
    metadata = shm.get_metadata()
    print(f"Frame {metadata.seq}: {metadata.min_temp:.1f}°C - {metadata.max_temp:.1f}°C")
    
    # Get YUYV frame
    yuyv_frame = shm.get_yuyv_frame()
    
    # Get temperature array (96x96, uint16)
    temp_array = shm.get_temperature_array()
    
    # Get temperature map in Celsius (96x96, float32)
    temp_celsius = shm.get_temperature_map_celsius()
    
    # Mark frame as read
    shm.mark_frame_read()

device.stop()
```

---

## 🧩 Command Line Interface

| Command                | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `pythermal-preview` | Live preview with temperature overlay           |

Example:

```bash
pythermal-preview
```

This will start the thermal device and display a live view window.

---

## 🧰 API Overview

| Class                 | Purpose                                         |
| --------------------- | ----------------------------------------------- |
| `ThermalDevice`       | Manages thermal camera initialization via subprocess and shared memory access |
| `ThermalSharedMemory` | Reads thermal data from shared memory (YUYV frames, temperature arrays, metadata) |
| `ThermalRecorder`     | Records raw and colored frames to files        |
| `ThermalLiveView`     | Displays live thermal imaging feed with OpenCV  |
| `FrameMetadata`       | Named tuple containing frame metadata (seq, flag, dimensions, temperatures) |

---

## 🧪 Requirements

* Python ≥ 3.9
* ARM Linux environment (Jetson / OrangePi / Raspberry Pi)
* NumPy, OpenCV (auto-installed via pip)
* Thermal camera connected via USB
* Proper USB permissions (set up via `setup.sh`)

---

## ⚙️ Architecture

### Native Runtime

The library uses a native binary (`pythermal-recorder`) that runs as a separate process and writes thermal data to shared memory (`/dev/shm/yuyv240_shm`). The Python library communicates with this process via shared memory for efficient zero-copy data access.

### Bundled Files

The package includes the following native files under `pythermal/_native/armLinux/`:

```
pythermal/_native/armLinux/
├── pythermal-recorder      # Main thermal recorder executable
├── libHCUSBSDK.so            # Hikvision USB SDK library
├── libhpr.so                 # Hikvision processing library
├── libusb-1.0.so*            # USB library dependencies
└── libuvc.so                  # UVC library
```

### Shared Memory Layout

The shared memory (`/dev/shm/yuyv240_shm`) contains:

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
  - Verify USB permissions are set up correctly (run `setup.sh`)
  - Try disconnecting and reconnecting the camera
  - Check that no other process is using the thermal camera

* **`RuntimeError: Thermal recorder process exited unexpectedly`**
  Check the process output for error messages. Common issues:
  - Camera not detected
  - Missing USB permissions
  - Missing shared libraries (check `LD_LIBRARY_PATH`)

---

## 📦 Directory Structure

```
pythermal-devkit/
├── pythermal/
│   ├── __init__.py
│   ├── device.py              # ThermalDevice class (manages subprocess)
│   ├── thermal_shared_memory.py  # Shared memory reader
│   ├── record.py              # ThermalRecorder class
│   ├── live_view.py           # ThermalLiveView class
│   └── _native/
│       └── armLinux/
│           ├── pythermal-recorder
│           └── *.so            # Native libraries
├── setup.sh                   # Setup script for permissions and compilation
├── setup-thermal-permissions.sh
├── setup.py                   # Python package setup
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
**📧 Device Access:** thermal@thingx-tech.com