Architecture
============

Native Runtime
--------------

The library uses a native binary (`pythermal-recorder`) that runs as a separate process and writes thermal data to shared memory (`/dev/shm/yuyv240_shm`). The Python library communicates with this process via shared memory for efficient zero-copy data access.

Shared Memory Layout
--------------------

The shared memory (`/dev/shm/yuyv240_shm`) contains:

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

The `ThermalDevice` class:

1. Starts `pythermal-recorder` as a subprocess
2. Waits for shared memory to become available
3. Provides access to thermal data via `ThermalSharedMemory`
4. Automatically cleans up the process on exit

Module Structure
----------------

.. code-block:: text

   pythermal/
   ├── device.py              # ThermalDevice class
   ├── thermal_shared_memory.py  # Shared memory reader
   ├── record.py              # ThermalRecorder class
   ├── live_view.py           # ThermalLiveView class
   └── detections/
       ├── utils.py           # Shared utilities
       ├── temperature_detection.py  # Temperature-based detection
       ├── motion_detection.py      # Background subtraction
       └── roi.py             # ROI management

