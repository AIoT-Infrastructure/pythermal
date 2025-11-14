#!/usr/bin/env python3
"""
Thermal Recorder

Records thermal camera data to files using the shared memory interface.
"""

import struct
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from .device import ThermalDevice
from .thermal_shared_memory import ThermalSharedMemory, WIDTH, HEIGHT, TEMP_WIDTH, TEMP_HEIGHT


class ThermalRecorder:
    """
    Records thermal camera data to files.
    
    Records both YUYV frames and temperature arrays with timestamps.
    """
    
    def __init__(self, output_dir: str = "recordings", color: bool = True):
        """
        Initialize thermal recorder.
        
        Args:
            output_dir: Directory to save recordings
            color: If True, also record colored RGB frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color = color
        self.device: Optional[ThermalDevice] = None
        self._device_owned = False  # Track if we created the device
        self.recording = False
        self.frame_count = 0
        self.start_time: Optional[float] = None
        
    def start(self, device: Optional[ThermalDevice] = None) -> bool:
        """
        Start recording.
        
        Args:
            device: Optional ThermalDevice instance. If None, creates a new one.
            
        Returns:
            True if successful, False otherwise
        """
        if self.recording:
            return True
        
        if device is None:
            self.device = ThermalDevice()
            self._device_owned = True
            if not self.device.start():
                return False
        else:
            self.device = device
            self._device_owned = False
            if not self.device.is_running():
                if not self.device.start():
                    return False
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"thermal_{timestamp}.tseq"
        
        # Open output file for writing
        self._file_handle = open(self.output_file, "wb")
        
        # Write header: format version, color flag
        header = b"TSEQ\x01" + (b"\x01" if self.color else b"\x00")
        self._file_handle.write(header)
        
        return True
    
    def stop(self):
        """Stop recording and close output file."""
        if not self.recording:
            return
        
        self.recording = False
        
        if hasattr(self, "_file_handle") and self._file_handle:
            self._file_handle.close()
        
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"Recording stopped: {self.frame_count} frames in {duration:.2f}s")
        print(f"Saved to: {self.output_file}")
    
    def record_frame(self) -> bool:
        """
        Record a single frame from shared memory.
        
        Returns:
            True if frame was recorded, False otherwise
        """
        if not self.recording or self.device is None:
            return False
        
        if not self.device.is_running():
            return False
        
        shm = self.device.get_shared_memory()
        
        # Wait for new frame
        if not shm.has_new_frame():
            return False
        
        # Get metadata
        metadata = shm.get_metadata()
        if not metadata:
            return False
        
        # Get frame data
        yuyv = shm.get_yuyv_frame()
        temp_array = shm.get_temperature_array()
        
        if yuyv is None or temp_array is None:
            return False
        
        # Write frame data
        timestamp = time.time()
        
        # Frame header: timestamp (8 bytes), seq (4 bytes), metadata (12 bytes)
        frame_header = struct.pack("dIfff", timestamp, metadata.seq,
                                  metadata.min_temp, metadata.max_temp, metadata.avg_temp)
        self._file_handle.write(frame_header)
        
        # Write YUYV data
        self._file_handle.write(yuyv.tobytes())
        
        # Write temperature array
        self._file_handle.write(temp_array.tobytes())
        
        # Write colored frame if requested
        if self.color:
            # Convert YUYV to RGB
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
            self._file_handle.write(rgb.tobytes())
        
        # Mark frame as read
        shm.mark_frame_read()
        
        self.frame_count += 1
        return True
    
    def record_loop(self, duration: Optional[float] = None):
        """
        Record frames in a loop.
        
        Args:
            duration: Optional duration in seconds. If None, records until stopped.
        """
        if not self.recording:
            if not self.start():
                return
        
        end_time = time.time() + duration if duration else None
        
        try:
            while self.recording:
                if end_time and time.time() >= end_time:
                    break
                
                self.record_frame()
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            self.stop()
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        # Cleanup device only if we created it
        if self.device is not None and self._device_owned:
            self.device.cleanup()
    
    @staticmethod
    def replay(file_path: str, view_mode: str = 'yuyv', fps: Optional[float] = None):
        """
        Replay a recorded thermal camera file.
        
        Args:
            file_path: Path to the .tseq recording file
            view_mode: 'yuyv' or 'temperature' view mode
            fps: Target FPS for playback. If None, uses original timestamps.
        """
        from .live_view import ThermalLiveView
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        
        # Open file for reading
        with open(file_path, "rb") as f:
            # Read header: "TSEQ" + version (1 byte) + color flag (1 byte)
            header = f.read(6)
            if len(header) != 6 or header[:4] != b"TSEQ":
                print(f"Error: Invalid file format: {file_path}")
                return
            
            version = header[4]
            has_color = header[5] == 1
            
            print(f"Replaying: {file_path}")
            print(f"Format version: {version}, Color: {has_color}")
            print("Press 'q' to quit, 't' to toggle view mode")
            
            # Calculate frame sizes
            yuyv_size = WIDTH * HEIGHT * 2  # 115200 bytes
            temp_size = TEMP_WIDTH * TEMP_HEIGHT * 2  # 18432 bytes
            rgb_size = WIDTH * HEIGHT * 3 if has_color else 0  # 172800 bytes if color
            frame_header_size = 24  # timestamp (8) + seq (4) + 3 floats (12)
            frame_size = frame_header_size + yuyv_size + temp_size + rgb_size
            
            # Create a replay viewer (without device)
            viewer = ThermalLiveView(device=None)
            viewer.view_mode = view_mode
            
            # Initialize display window
            window_name = "Thermal Camera Replay"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 480, 640)
            
            # Set mouse callback
            cv2.setMouseCallback(window_name, viewer.mouse_callback)
            
            frame_count = 0
            last_timestamp = None
            last_display_time = time.time()
            
            try:
                while True:
                    # Read frame header
                    frame_header = f.read(frame_header_size)
                    if len(frame_header) < frame_header_size:
                        print(f"\nEnd of file reached. Total frames: {frame_count}")
                        break
                    
                    # Unpack frame header
                    timestamp, seq, min_temp, max_temp, avg_temp = struct.unpack("dIfff", frame_header)
                    
                    # Read YUYV data
                    yuyv_bytes = f.read(yuyv_size)
                    if len(yuyv_bytes) < yuyv_size:
                        break
                    yuyv = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
                    
                    # Read temperature array
                    temp_bytes = f.read(temp_size)
                    if len(temp_bytes) < temp_size:
                        break
                    temp_array = np.frombuffer(temp_bytes, dtype=np.uint16).reshape((TEMP_HEIGHT, TEMP_WIDTH))
                    
                    # Read RGB if present (but we'll use YUYV for display)
                    if has_color:
                        rgb_bytes = f.read(rgb_size)
                        if len(rgb_bytes) < rgb_size:
                            break
                    
                    # Store temperature data for mouse callback
                    viewer.current_temp_data = temp_array.copy()
                    
                    # Display based on view mode
                    if viewer.view_mode == 'yuyv':
                        thermal_image = viewer.get_original_yuyv(yuyv)
                    else:  # temperature view
                        thermal_image = viewer.get_temperature_view(temp_array, min_temp, max_temp)
                    
                    # Calculate FPS (increment frame_count first, then calculate)
                    frame_count += 1
                    viewer.frame_count += 1
                    current_fps = viewer.calculate_fps()
                    
                    # Draw overlay
                    thermal_image = viewer.draw_overlay(
                        thermal_image, min_temp, max_temp, avg_temp, seq, current_fps
                    )
                    
                    # Display image
                    cv2.imshow(window_name, thermal_image)
                    
                    # Handle timing
                    if fps is not None:
                        # Use fixed FPS
                        target_delay = 1.0 / fps
                        elapsed = time.time() - last_display_time
                        if elapsed < target_delay:
                            time.sleep(target_delay - elapsed)
                        last_display_time = time.time()
                    elif last_timestamp is not None:
                        # Use original timestamps
                        frame_delay = timestamp - last_timestamp
                        if frame_delay > 0:
                            time.sleep(frame_delay)
                    last_timestamp = timestamp
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        viewer.view_mode = 'temperature' if viewer.view_mode == 'yuyv' else 'yuyv'
                        print(f"Switched to {viewer.view_mode.upper()} view")
                    
            except KeyboardInterrupt:
                print("\nReplay interrupted by user")
            except Exception as e:
                print(f"Error during replay: {e}")
                import traceback
                traceback.print_exc()
            finally:
                cv2.destroyAllWindows()
                print("Replay stopped")

