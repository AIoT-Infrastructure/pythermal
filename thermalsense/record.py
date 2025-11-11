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

