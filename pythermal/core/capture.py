#!/usr/bin/env python3
"""
Thermal Capture - Unified Interface

Provides a unified interface for thermal camera capture (live or recorded).
Similar to cv2.VideoCapture, accepts file paths for recorded sequences or
0/None/empty string for live camera.

Provides the same interface as ThermalSharedMemory, allowing detection modules
to work with both live and recorded data without modification.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
from .thermal_shared_memory import FrameMetadata, WIDTH, HEIGHT
from .sequence_reader import ThermalSequenceReader


class ThermalCapture:
    """
    Unified interface for thermal camera capture (live or recorded)
    
    Similar to cv2.VideoCapture, accepts:
    - File path (str): Opens recorded .tseq file
    - 0, None, or empty string: Uses live camera
    
    Provides the same interface as ThermalSharedMemory, allowing detection modules
    to work with both live and recorded data without modification.
    """
    
    def __init__(self, source: Union[str, int, None] = None, device_index: Optional[int] = None, native_dir: Optional[str] = None):
        """
        Initialize thermal capture from source
        
        Args:
            source: File path for recorded .tseq file, or 0/None/empty string for live camera
                   If None, defaults to live camera (0)
            device_index: Index of the USB device to use (0 for first device, 1 for second, etc.).
                         Default is 0. Only used for live camera. Each device uses a separate shared memory segment.
            native_dir: Optional path to native directory containing pythermal-recorder.
                       If None, uses default package location. Only used for live camera.
        
        Raises:
            FileNotFoundError: If file path doesn't exist
            ValueError: If file format is invalid
            RuntimeError: If thermal camera is not available
            TimeoutError: If thermal camera initialization times out
        """
        from .device import ThermalDevice
        
        # Normalize source: None, 0, "0", or "" means live camera
        if source is None or source == 0 or source == "0" or source == "":
            # Use live camera
            self._is_recorded = False
            self._device = ThermalDevice(native_dir=native_dir, device_index=device_index)
            
            # Start device and initialize shared memory
            if not self._device.start():
                raise RuntimeError(
                    "Failed to start thermal device. "
                    "Make sure the thermal camera is connected and permissions are set up."
                )
            
            self._data_source = self._device.get_shared_memory()
            
            if not self._data_source.initialize():
                self._device.stop()
                raise RuntimeError(
                    "Failed to initialize shared memory. "
                    "Thermal camera may not be available."
                )
            
            # Verify we can read metadata
            metadata = self._data_source.get_metadata()
            if metadata is None:
                self._device.stop()
                raise RuntimeError(
                    "Failed to read metadata from thermal camera. "
                    "Camera may not be ready."
                )
            
            self._file_path = None
            self._total_frames = None
            
        else:
            # Use recorded file
            self._is_recorded = True
            self._device = None
            file_path = Path(source)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {source}")
            
            # Create sequence reader
            self._data_source = ThermalSequenceReader(str(file_path))
            self._file_path = file_path
            self._total_frames = self._data_source._total_frames
    
    @property
    def is_recorded(self) -> bool:
        """Check if source is a recorded file"""
        return self._is_recorded
    
    @property
    def is_live(self) -> bool:
        """Check if source is live camera"""
        return not self._is_recorded
    
    def get_metadata(self) -> Optional[FrameMetadata]:
        """
        Get current frame metadata
        
        Returns:
            FrameMetadata named tuple, or None if no frame available
        """
        return self._data_source.get_metadata()
    
    def has_new_frame(self) -> bool:
        """
        Check if a new frame is available
        
        Returns:
            True if new frame available, False otherwise
        """
        return self._data_source.has_new_frame()
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame (VideoCapture-like interface)
        
        Returns:
            Tuple of (success, frame) where frame is BGR image (240x240x3)
        """
        if self._is_recorded:
            return self._data_source.read()
        else:
            # For live data, convert YUYV to BGR
            if not self.has_new_frame():
                return False, None
            
            yuyv = self._data_source.get_yuyv_frame()
            if yuyv is None:
                return False, None
            
            import cv2
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            return True, bgr
    
    def get_yuyv_frame(self) -> Optional[np.ndarray]:
        """
        Get current YUYV frame (ThermalSharedMemory interface)
        
        Returns:
            YUYV frame array (240x240x2), or None if no frame available
        """
        return self._data_source.get_yuyv_frame()
    
    def get_temperature_array(self) -> Optional[np.ndarray]:
        """
        Get current temperature array (ThermalSharedMemory interface)
        
        Returns:
            Temperature array (96x96) with uint16 values, or None if no frame available
        """
        return self._data_source.get_temperature_array()
    
    def mark_frame_read(self):
        """
        Mark current frame as read (ThermalSharedMemory interface)
        """
        self._data_source.mark_frame_read()
    
    def set(self, prop: int, value: float) -> bool:
        """
        Set property (VideoCapture-like interface)
        
        Args:
            prop: Property ID (e.g., cv2.CAP_PROP_POS_FRAMES)
            value: Property value
            
        Returns:
            True if successful, False otherwise
        """
        if self._is_recorded:
            return self._data_source.set(prop, value)
        else:
            # Live camera doesn't support seeking
            return False
    
    def get(self, prop: int) -> float:
        """
        Get property (VideoCapture-like interface)
        
        Args:
            prop: Property ID (e.g., cv2.CAP_PROP_POS_FRAMES, cv2.CAP_PROP_FRAME_COUNT)
            
        Returns:
            Property value
        """
        if self._is_recorded:
            return self._data_source.get(prop)
        else:
            # Live camera properties
            import cv2
            
            if prop == cv2.CAP_PROP_POS_FRAMES:
                # Live camera doesn't have frame index
                return 0.0
            elif prop == cv2.CAP_PROP_FRAME_COUNT:
                # Live camera has infinite frames
                return 0.0
            elif prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(WIDTH)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(HEIGHT)
            elif prop == cv2.CAP_PROP_FPS:
                # Unknown for live camera
                return 0.0
            
            return 0.0
    
    def isOpened(self) -> bool:
        """
        Check if capture is opened (VideoCapture-like interface)
        
        Returns:
            True if opened, False otherwise
        """
        if self._is_recorded:
            return self._data_source.isOpened()
        else:
            return self._device.is_running() if self._device else False
    
    def is_initialized(self) -> bool:
        """Check if capture is initialized"""
        if self._is_recorded:
            return self._data_source.is_initialized()
        else:
            return self._data_source.is_initialized() if self._data_source else False
    
    def release(self):
        """
        Release resources (VideoCapture-like interface)
        """
        if self._is_recorded:
            self._data_source.release()
        else:
            if self._device:
                self._device.stop()
                self._device = None
            self._data_source = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()

