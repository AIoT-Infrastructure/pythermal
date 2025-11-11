# 🔥 Thermal Sensing Library

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

* **Thermal Analytics**
  Built-in lightweight models for:

  * Human body detection
  * Skeleton (pose) detection
  * ROI temperature statistics (max / min / avg)

* **Offline Replay and Analysis**
  Replay recorded sessions for algorithm benchmarking or dataset generation.

---

## 🚀 Installation

Install directly on an ARM Linux device (e.g., Jetson, OrangePi, Raspberry Pi):

```bash
uv pip install thermal-sensing
```

Or from source:

```bash
git clone https://github.com/<your-org>/thermal-sensing.git
cd thermal-sensing
uv pip install .
```

> ✅ **Bundled Native Runtime**
> The package ships with a prebuilt native library (e.g., `libhikiface_aarch64.so`) under `thermalsense/_native/`.
> You do **not** need to install vendor headers or a separate SDK at runtime.
> Only the compiled `.so` is exposed; no Hikvision source or headers are distributed.

---

## 🧠 Quick Start

### 1. Record Thermal Frames

```python
from thermalsense import ThermalRecorder

rec = ThermalRecorder(output="session.tseq", color=True)
rec.start()              # start recording frames
time.sleep(10)
rec.stop()               # stop recording
```

This records both:

* Raw temperature frames (`96×96`, uint16)
* Colored visual frames (`240×240`, uint8 RGB)

---

### 2. Live Analytics

```python
from thermalsense import ThermalAnalyzer

analyzer = ThermalAnalyzer(model="person_pose.onnx")
analyzer.start()

for frame in analyzer.stream():
    results = analyzer.recognize(frame)
    print("Detected:", results['people'], "Pose:", results['pose'])
```

Outputs real-time recognition results including detected persons and skeletal keypoints.

---

### 3. Replay a Recorded Session

```python
from thermalsense import ThermalReplay

replayer = ThermalReplay("session.tseq")
for frame in replayer:
    temp = frame.temperature_map()
    print("Max:", temp.max(), "Min:", temp.min())
```

---

## 🎨 Color Palettes

Built-in palettes include:

* `white-hot`
* `black-hot`
* `ironbow`
* `rainbow`
* `arctic`
* `sepia`

You can also import custom LUTs:

```python
from thermalsense.color import load_palette
load_palette("my_palette.csv", name="mycool")
```

---

## 🧩 Command Line Interface

| Command                | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `thermalsense-preview` | Live preview with color map and temperature bar |
| `thermalsense-record`  | Record session with metadata                    |
| `thermalsense-replay`  | Replay `.tseq` file                             |
| `thermalsense-infer`   | Run person and pose detection                   |

Example:

```bash
thermalsense-infer --model person_pose.onnx --device /dev/ttyUSB0 --palette ironbow
```

---

## 🧰 API Overview

| Class             | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `ThermalDevice`   | Connects to and streams from the thermal camera |
| `ThermalRecorder` | Records raw and colored frames                  |
| `ThermalReplay`   | Replays recorded data                           |
| `ThermalAnalyzer` | Provides live analytics and recognition         |
| `Colorizer`       | Converts 16-bit thermal maps to RGB             |

---

## 🧪 Requirements

* Python ≥ 3.9
* Arm Linux environment (Jetson / OrangePi / Raspberry Pi)
* NumPy, OpenCV, onnxruntime (auto-installed)
* No external SDK installation is required at runtime (the compiled `.so` is bundled)

---

## ⚙️ Bundled Native Runtime (.so)

### Layout & Loading

The wheel includes one of the following (per-arch) under your package:

```
thermalsense/_native/
└── libhikiface_aarch64.so     # typical for modern ARM64 boards
# or
└── libhikiface_armv7.so       # if you also publish ARMv7 wheels
```

At import, the library loader resolves in this order:

1. **Packaged path**: `thermalsense/_native/libhikiface_<arch>.so`
2. **Override (optional)**: If you set `THERMALSENSE_LIB_DIR`, it will try
   `$THERMALSENSE_LIB_DIR/libhikiface_<arch>.so`
3. **System paths** (discouraged): `/usr/local/lib`, `LD_LIBRARY_PATH`

> If you maintain multiple boards, publish separate wheels per-arch (e.g., `manylinux2014_aarch64`).

### Exposed C ABI (stable)

The native `.so` exports a minimal C interface used by the Python wrapper.
It does **not** expose any vendor SDK symbols or headers.

#### Data structures

```c
#define TEMP_DATA_SIZE (TEMP_WIDTH * TEMP_HEIGHT * 2) /* 96x96 uint16 */

typedef struct {
    unsigned char data[FRAME_SIZE];         /* raw YUYV bytes */
    unsigned char temp_data[TEMP_DATA_SIZE];/* 96x96, uint16_t */
    unsigned int  seq;                      /* increasing sequence */
    unsigned int  flag;                     /* 1=new, 0=consumed */
    unsigned int  width;                    /* YUYV width  */
    unsigned int  height;                   /* YUYV height */
    float         min_temp;                 /* °C */
    float         max_temp;                 /* °C */
    float         avg_temp;                 /* °C */
    unsigned char reserved[4];
} FrameBuf;

typedef struct TsHandle_ TsHandle;

typedef struct {
    char serial[64];
    char model[64];
    int  width;     /* e.g., 240 */
    int  height;    /* e.g., 240 */
    int  t_width;   /* 96 */
    int  t_height;  /* 96 */
} TsDeviceInfo;
```

#### Functions

```c
// Initialization / shutdown
TsStatus ts_init(void);
void     ts_shutdown(void);

// Device enumeration / lifecycle
int      ts_list_devices(TsDeviceInfo* out, int max_out);
TsStatus ts_open(const char* serial, TsHandle** out); // serial=NULL -> first device
void     ts_close(TsHandle* h);

// Streaming
TsStatus ts_start_stream(TsHandle* h);
void     ts_stop_stream(TsHandle* h);

// Frame acquisition (zero-copy ring buffer)
TsStatus ts_acquire_frame(TsHandle* h, FrameBuf** out, int timeout_ms, int peek);
void     ts_release_frame(TsHandle* h, FrameBuf* fb);

// Configuration
TsStatus ts_set_emissivity(TsHandle* h, float epsilon);
TsStatus ts_set_distance_m(TsHandle* h, float meters);
TsStatus ts_set_ambient_c(TsHandle* h, float ambient_c);
TsStatus ts_trigger_nuc(TsHandle* h);

// Utilities (optional)
TsStatus ts_colorize_temp(const unsigned short* temp96x96, int t_w, int t_h,
                          unsigned char* rgb_out, int out_w, int out_h,
                          const char* palette, float tmin_c, float tmax_c);

TsStatus ts_roi_stats(const unsigned short* temp96x96, int x, int y, int w, int h,
                      float* min_c, float* max_c, float* avg_c);

const char* ts_version(void);
```

**Design notes**

* Pure **C ABI** (no C++ name mangling).
* **Zero-copy** access to `temp_data`/`data` via ring buffer; `flag` prevents reprocessing.
* The Python wrapper (ctypes/pybind11) maps buffers into NumPy arrays directly.
* Internals may dynamically link vendor bits; none are exposed.

### Troubleshooting

* **`OSError: cannot load library`**
  Ensure your wheel matches the board architecture (e.g., `aarch64`).
  If you built from source, confirm the `.so` exists at `thermalsense/_native/` inside the installed site-packages.
  Optionally set:

  ```bash
  export THERMALSENSE_LIB_DIR=/opt/thermalsense/native
  ```
* **Mismatched ABI**
  If Python warns about `ts_version()` mismatch, upgrade the package:

  ```bash
  uv pip install --upgrade thermal-sensing
  ```

---

## 📦 Directory Structure

```
thermal-sensing/
├── thermalsense/
│   ├── _native/
│   │   └── libhikiface_aarch64.so
│   ├── device.py
│   ├── record.py
│   ├── replay.py
│   ├── color.py
│   ├── models/
│   └── utils/
├── examples/
│   ├── record_and_preview.py
│   ├── live_analytics.py
│   └── replay_analysis.py
└── README.md
```

---

## 📄 License

This library is released under the **Apache 2.0 License** for research and non-commercial use.
Only the compiled native library (`.so`) is shipped; no vendor source or headers are distributed.

---

## 💡 Acknowledgements

Developed by **AIoT Lab, CUHK**
Lead Contributor: [Yunqi Guo](mailto:yunqiguo@cuhk.edu.hk)
