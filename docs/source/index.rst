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

   from pythermal import ThermalDevice, detect_object_centers

   # Initialize thermal device
   device = ThermalDevice()
   device.start()
   shm = device.get_shared_memory()

   # Detect objects
   if shm.has_new_frame():
       metadata = shm.get_metadata()
       temp_array = shm.get_temperature_array()
       
       objects = detect_object_centers(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           temp_min=31.0,
           temp_max=39.0
       )
       
       print(f"Detected {len(objects)} objects")

   device.stop()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
