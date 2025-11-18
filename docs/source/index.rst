PyThermal Documentation
=========================

**A lightweight Python library for thermal sensing and analytics.**

PyThermal provides unified APIs for recording, visualization, and intelligent analysis of thermal data from Hikvision or compatible infrared sensors.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples
   architecture

Features
--------

* **Raw Frame Recording** - Capture and store radiometric thermal frames with timestamps
* **Colored Visualization** - Generate pseudo-color thermal images with adjustable color maps
* **Live Stream Interface** - Stream frames in real time with temperature conversion
* **Shared Memory Architecture** - Efficient zero-copy access to thermal data
* **Thermal Object Detection** - Detect objects based on temperature ranges
* **Motion Detection** - Background subtraction for detecting moving objects
* **ROI Management** - Zone monitoring with multiple regions of interest
* **Shape Analysis** - Filter objects by shape characteristics

Quick Start
-----------

.. code-block:: python

   from pythermal import ThermalCapture, detect_humans_adaptive

   # Initialize thermal capture (works for both live and recorded data)
   capture = ThermalCapture()  # Use None or 0 for live camera, or file path for recorded

   # Detect objects using adaptive human detection
   if capture.has_new_frame():
       metadata = capture.get_metadata()
       temp_array = capture.get_temperature_array()
       
       objects = detect_humans_adaptive(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           environment_temp=None,  # Auto-estimate from frame, or provide value (e.g., 22.0)
           min_area=50,
           min_temp_above_env=2.0,
           max_temp_limit=42.0
       )
       
       print(f"Detected {len(objects)} objects")
       capture.mark_frame_read()

   capture.release()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
