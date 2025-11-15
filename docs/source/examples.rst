Examples
========

PyThermal includes several example scripts demonstrating different features.

Basic Examples
--------------

* **Live View** (`examples/live_view.py`) - Display real-time thermal feed with multiple view modes (YUYV, temperature, temperature Celsius)
* **Record Thermal** (`examples/record_thermal.py`) - Record thermal frames to files (MP4, raw data, or both)
* **Detect Objects** (`examples/detect_objects.py`) - Detect and visualize objects based on temperature ranges
* **Motion Detection** (`examples/detect_motion.py`) - Detect moving objects using background subtraction
* **ROI Monitoring** (`examples/detect_roi.py`) - Zone monitoring with regions of interest (ROIs)

Advanced Examples
------------------

* **YOLO Object Detection** (`examples/yolo_object_detection.py`) - Detect objects using YOLO v11 models
* **YOLO Pose Detection** (`examples/yolo_pose_detection.py`) - Detect human poses and keypoints using YOLO v11

Running Examples
----------------

Basic examples can be run directly:

.. code-block:: bash

   # Live view (supports both live camera and recorded files)
   python examples/live_view.py
   python examples/live_view.py recordings/thermal_20240101.tseq

   # Record thermal data
   python examples/record_thermal.py --duration 10 --format both

   # Object detection
   python examples/detect_objects.py

   # Motion detection
   python examples/detect_motion.py

   # ROI monitoring
   python examples/detect_roi.py

YOLO examples require the `ultralytics` package:

.. code-block:: bash

   # Install YOLO support
   pip install pythermal[yolo]
   # or
   pip install ultralytics>=8.0.0

   # Run YOLO examples
   python examples/yolo_object_detection.py
   python examples/yolo_pose_detection.py

All examples support both live camera feeds and recorded sequences using the `ThermalCapture` interface.

See the `examples/README.md` file for detailed usage instructions.

