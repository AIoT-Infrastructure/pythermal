Quick Start Guide
=================

This guide will help you get started with PyThermal quickly.

Unified Capture Interface
-------------------------

The `ThermalCapture` class provides a unified interface for both live camera feeds and recorded sequences. It's similar to `cv2.VideoCapture` and automatically handles device initialization:

.. code-block:: python

   from pythermal import ThermalCapture

   # For live camera (default)
   capture = ThermalCapture()  # or ThermalCapture(0) or ThermalCapture(None)

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

Or use as a context manager:

.. code-block:: python

   with ThermalCapture() as capture:
       if capture.has_new_frame():
           metadata = capture.get_metadata()
           yuyv_frame = capture.get_yuyv_frame()
           temp_array = capture.get_temperature_array()
           capture.mark_frame_read()
       # Automatically releases on exit

Live View
---------

Display real-time thermal imaging feed or replay recorded sequences:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from pythermal import ThermalLiveView

   # Live camera view
   viewer = ThermalLiveView()
   viewer.run()  # Opens OpenCV window with live thermal feed

   # Or replay a recorded file
   viewer = ThermalLiveView(source="recordings/thermal_20240101.tseq")
   viewer.run()

Enhanced Live View with Multiple View Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example ``live_view.py`` demonstrates an enhanced live view with multiple display modes:

.. code-block:: python

   from pythermal import ThermalLiveView
   from pythermal.utils import estimate_environment_temperature_v1

   class EnhancedLiveView(ThermalLiveView):
       """Enhanced live view with additional color modes"""
       
       def __init__(self, source=None):
           super().__init__(source)
           # View modes: 'yuyv', 'temperature', 'temperature_celsius'
           self.view_modes = ['yuyv', 'temperature', 'temperature_celsius']
           self.view_mode_index = 0
           self.view_mode = self.view_modes[self.view_mode_index]
           self.clahe_contrast = 3.0  # Contrast enhancement level
       
       def toggle_view_mode(self):
           """Toggle to next view mode"""
           self.view_mode_index = (self.view_mode_index + 1) % len(self.view_modes)
           self.view_mode = self.view_modes[self.view_mode_index]
           print(f"Switched to {self.view_mode.upper()} view")
       
       def adjust_contrast(self, delta: float):
           """Adjust CLAHE contrast level"""
           self.clahe_contrast = max(1.0, min(8.0, self.clahe_contrast + delta))
           print(f"Contrast level: {self.clahe_contrast:.2f}")

   # Use enhanced viewer
   viewer = EnhancedLiveView()
   viewer.run()

Keyboard Controls
~~~~~~~~~~~~~~~~~

- ``q`` - Quit
- ``t`` - Toggle view mode (YUYV / Temperature / Temperature Celsius)
- ``+`` or ``=`` - Increase contrast
- ``-`` or ``_`` - Decrease contrast
- Mouse hover - See temperature at cursor position
- ``SPACE`` - Pause/Resume (for recorded files only)

Command-Line Usage
~~~~~~~~~~~~~~~~~~

Run the enhanced live view example from command line:

.. code-block:: bash

   # Live view (default)
   python examples/live_view.py

   # Recorded file replay
   python examples/live_view.py recordings/thermal_20240101.tseq

   # Recorded with custom FPS
   python examples/live_view.py file.tseq --fps 30

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

   from pythermal import ThermalCapture, detect_object_centers

   capture = ThermalCapture()  # Live camera, or pass file path for recorded data

   if capture.has_new_frame():
       metadata = capture.get_metadata()
       temp_array = capture.get_temperature_array()
       
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
       
       capture.mark_frame_read()

   capture.release()

Motion Detection
----------------

Detect moving objects using background subtraction:

.. code-block:: python

   from pythermal import ThermalCapture, BackgroundSubtractor, detect_moving_objects

   capture = ThermalCapture()  # Live camera, or pass file path for recorded data

   # Initialize background subtractor
   bg_subtractor = BackgroundSubtractor(learning_rate=0.01)

   if capture.has_new_frame():
       metadata = capture.get_metadata()
       temp_array = capture.get_temperature_array()
       
       # Detect moving objects
       moving_objects, foreground_mask = detect_moving_objects(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           background_subtractor=bg_subtractor,
           temp_threshold=2.0
       )
       
       print(f"Detected {len(moving_objects)} moving objects")
       
       capture.mark_frame_read()

   capture.release()

ROI Zone Monitoring
-------------------

Monitor specific regions of interest:

.. code-block:: python

   from pythermal import ThermalCapture, ROIManager, detect_object_centers

   capture = ThermalCapture()  # Live camera, or pass file path for recorded data

   # Initialize ROI manager
   roi_manager = ROIManager(image_width=96, image_height=96)
   
   # Add center 30x30 ROI
   roi_manager.add_center_roi(
       size=30,
       name="Center",
       temp_min=30.0,
       temp_max=39.0
   )

   if capture.has_new_frame():
       metadata = capture.get_metadata()
       temp_array = capture.get_temperature_array()
       
       # Detect all objects
       all_objects = detect_object_centers(
           temp_array=temp_array,
           min_temp=metadata.min_temp,
           max_temp=metadata.max_temp,
           temp_min=30.0,
           temp_max=39.0
       )
       
       # Filter by ROI
       roi_objects = roi_manager.filter_objects_by_roi(all_objects)
       
       print(f"Objects in ROI: {len(roi_objects)}")
       
       capture.mark_frame_read()

   capture.release()

Advanced: Direct Shared Memory Access
--------------------------------------

For advanced use cases, you can access the shared memory interface directly:

.. code-block:: python

   from pythermal import ThermalDevice, ThermalSharedMemory

   device = ThermalDevice()
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

