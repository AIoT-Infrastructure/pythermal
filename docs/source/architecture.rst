Architecture
============

Unified Capture Interface
--------------------------

The library provides a unified interface (`ThermalCapture`) that works seamlessly with both live camera feeds and recorded sequences. This allows detection modules and applications to work with either source without modification.

- **Live Camera**: Uses `ThermalDevice` and `ThermalSharedMemory` to access real-time thermal data
- **Recorded Sequences**: Uses `ThermalSequenceReader` to read pre-recorded `.tseq` files
- **Unified API**: `ThermalCapture` automatically selects the appropriate backend based on the source

Native Runtime
--------------

For live camera access, the library uses a native binary (`pythermal-recorder`) that runs as a separate process and writes thermal data to shared memory. Each device uses its own shared memory segment for parallel operation. The Python library communicates with these processes via shared memory for efficient zero-copy data access.

Multi-Device Architecture
--------------------------

PyThermal supports multiple thermal cameras connected simultaneously:

- **Device Enumeration**: Uses `enumerate_devices` helper executable to query USB SDK for connected devices
- **Consistent ID Mapping**: Maps USB serial numbers to consistent device IDs (0, 1, 2, ...) stored in `~/.pythermal/device_mapping.json`
- **Device-Specific Shared Memory**: Each device uses a separate shared memory segment:
  - Device 0: `/dev/shm/yuyv240_shm`
  - Device 1: `/dev/shm/yuyv240_shm_1`
  - Device 2: `/dev/shm/yuyv240_shm_2`
  - etc.
- **Parallel Operation**: Multiple `ThermalDevice` instances can run simultaneously, each accessing its own shared memory segment

Shared Memory Layout
--------------------

Each shared memory segment (`/dev/shm/yuyv240_shm` or `/dev/shm/yuyv240_shm_{device_id}`) contains:

.. code-block:: text

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

Process Management
------------------

For live camera feeds, the `ThermalDevice` class:

1. Enumerates devices via `DeviceManager` to map device_id to enum_index (SDK's internal index)
2. Starts `pythermal-recorder` as a subprocess with both enum_index (for device selection) and device_id (for shared memory naming)
3. Waits for shared memory to become available
4. Provides access to thermal data via `ThermalSharedMemory`
5. Automatically cleans up the process on exit

The `ThermalCapture` class automatically manages `ThermalDevice` for live sources or `ThermalSequenceReader` for recorded files, providing a unified interface.

**Device Selection Flow:**

1. User specifies `device_index` (consistent ID) or `None` (auto-select smallest)
2. `DeviceManager` maps `device_id` to USB serial number via `device_mapping.json`
3. `DeviceManager` enumerates current devices and matches serial number to `enum_index` (SDK's internal index)
4. `ThermalDevice` passes `enum_index` to C++ recorder for device selection
5. `ThermalDevice` passes `device_id` to C++ recorder for shared memory naming
6. C++ recorder creates device-specific shared memory segment
7. Python reads from the correct shared memory segment based on `device_id`

Module Structure
----------------

.. code-block:: text

   pythermal/
   ├── __init__.py
   ├── core/                      # Core thermal camera components
   │   ├── __init__.py
   │   ├── device.py              # ThermalDevice class (manages subprocess)
   │   ├── thermal_shared_memory.py  # Shared memory reader
   │   ├── sequence_reader.py     # ThermalSequenceReader for recorded files
   │   └── capture.py             # ThermalCapture unified interface
   ├── record.py                  # ThermalRecorder class
   ├── live_view.py               # ThermalLiveView class
   ├── detections/                # Object detection module
   │   ├── __init__.py
   │   ├── utils.py               # Shared utilities and shape analysis
   │   ├── temperature_detection.py  # Temperature-based detection
   │   ├── motion_detection.py    # Background subtraction and motion detection
   │   ├── roi.py                 # ROI management and zone monitoring
   │   └── yolo/                  # YOLO v11 detection module
   │       ├── __init__.py
   │       ├── object_detection.py    # YOLO object detection
   │       ├── pose_detection.py      # YOLO pose detection
   │       └── models/                # Custom thermal models directory
   ├── utils/                     # Utility modules
   │   ├── __init__.py
   │   └── environment.py         # Environment temperature estimation
   ├── usb_setup/                 # USB setup scripts
   │   ├── setup.sh
   │   ├── setup-thermal-permissions.sh
   │   └── 99-thermal-camera.rules
   └── _native/                   # Native binaries and libraries
       ├── linux64/               # x86_64 binaries
       │   ├── pythermal-recorder
       │   └── *.so               # Native libraries
       └── armLinux/              # ARM binaries
           ├── pythermal-recorder
           └── *.so               # Native libraries

Bundled Native Runtime
-----------------------

The package includes native files for both x86_64 and ARM architectures:

**x86_64 (`pythermal/_native/linux64/`):**
- `pythermal-recorder` - Main thermal recorder executable
- `libHCUSBSDK.so` - Hikvision USB SDK library
- `libhpr.so` - Hikvision processing library
- `libusb-1.0.so*` - USB library dependencies
- `libuvc.so` - UVC library

**ARM (`pythermal/_native/armLinux/`):**
- `pythermal-recorder` - Main thermal recorder executable
- `libHCUSBSDK.so` - Hikvision USB SDK library
- `libhpr.so` - Hikvision processing library
- `libusb-1.0.so*` - USB library dependencies
- `libuvc.so` - UVC library

The library automatically detects your system architecture and loads the appropriate binaries.

