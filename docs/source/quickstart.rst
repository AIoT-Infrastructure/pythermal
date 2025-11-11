Quick Start Guide
=================

This guide will help you get started with PyThermal quickly.

Initialize Thermal Device
-------------------------

The `ThermalDevice` class manages the thermal camera initialization:

.. code-block:: python

   from pythermal import ThermalDevice

   # Create and start thermal device
   device = ThermalDevice()
   device.start()  # Starts pythermal-recorder subprocess

   # Access shared memory for reading thermal data
   shm = device.get_shared_memory()

   # When done, stop the device
   device.stop()

Or use as a context manager:

.. code-block:: python

   with ThermalDevice() as device:
       shm = device.get_shared_memory()
       # Use shared memory...
       # Device automatically stops on exit

Live View
---------

Display real-time thermal imaging feed:

.. code-block:: python

   from pythermal import ThermalLiveView

   viewer = ThermalLiveView()
   viewer.run()  # Opens OpenCV window with live thermal feed

Record Thermal Frames
---------------------

.. code-block:: python

   from pythermal import ThermalRecorder

   rec = ThermalRecorder(output_dir="recordings", color=True)
   rec.start()              # Starts device and begins recording
   rec.record_loop(duration=10)  # Record for 10 seconds
   rec.stop()               # Stop recording

Detect Objects
--------------

Detect objects based on temperature ranges:

.. code-block:: python

   from pythermal import ThermalDevice, detect_object_centers

   device = ThermalDevice()
   device.start()
   shm = device.get_shared_memory()

   if shm.has_new_frame():
       metadata = shm.get_metadata()
       temp_array = shm.get_temperature_array()
       
       # Detect objects in temperature range (default: 31-39°C for human body)
       objects = detect_object_centers(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           temp_min=31.0,
           temp_max=39.0
       )
       
       for obj in objects:
           print(f"Object at ({obj.center_x:.1f}, {obj.center_y:.1f}): "
                 f"{obj.avg_temperature:.1f}°C")

   device.stop()

Motion Detection
----------------

Detect moving objects using background subtraction:

.. code-block:: python

   from pythermal import ThermalDevice, BackgroundSubtractor, detect_moving_objects

   device = ThermalDevice()
   device.start()
   shm = device.get_shared_memory()

   # Initialize background subtractor
   bg_subtractor = BackgroundSubtractor(learning_rate=0.01)

   if shm.has_new_frame():
       metadata = shm.get_metadata()
       temp_array = shm.get_temperature_array()
       
       # Detect moving objects
       moving_objects, foreground_mask = detect_moving_objects(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           background_subtractor=bg_subtractor,
           temp_threshold=2.0
       )
       
       print(f"Detected {len(moving_objects)} moving objects")

   device.stop()

ROI Zone Monitoring
-------------------

Monitor specific regions of interest:

.. code-block:: python

   from pythermal import ThermalDevice, ROIManager, detect_object_centers

   device = ThermalDevice()
   device.start()
   shm = device.get_shared_memory()

   # Initialize ROI manager
   roi_manager = ROIManager(image_width=96, image_height=96)
   
   # Add center 30x30 ROI
   roi_manager.add_center_roi(
       size=30,
       name="Center",
       temp_min=30.0,
       temp_max=39.0
   )

   if shm.has_new_frame():
       metadata = shm.get_metadata()
       temp_array = shm.get_temperature_array()
       
       # Detect all objects
       all_objects = detect_object_centers(...)
       
       # Filter by ROI
       roi_objects = roi_manager.filter_objects_by_roi(all_objects)
       
       print(f"Objects in ROI: {len(roi_objects)}")

   device.stop()

