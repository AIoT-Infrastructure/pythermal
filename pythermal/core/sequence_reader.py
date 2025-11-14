#!/usr/bin/env python3
"""
Thermal Sequence Reader

Provides a VideoCapture-like interface for reading pre-recorded thermal camera sequences (.tseq files).
Compatible with the ThermalSharedMemory interface for seamless integration with detection modules.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Optional
from .thermal_shared_memory import FrameMetadata, WIDTH, HEIGHT, TEMP_WIDTH, TEMP_HEIGHT


class ThermalSequenceReader:
    """
    Reader for pre-recorded thermal camera sequences (.tseq files)
    
    Provides the same interface as ThermalSharedMemory, allowing detection modules
    to work with both live and recorded data without modification.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize thermal sequence reader
        
        Args:
            file_path: Path to the .tseq recording file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self._file_handle = None
        self._current_frame_data = None
        self._current_metadata = None
        self._frame_index = 0
        self._total_frames = 0
        self._has_color = False
        self._version = 0
        
        # Frame sizes
        self.yuyv_size = WIDTH * HEIGHT * 2  # 115200 bytes
        self.temp_size = TEMP_WIDTH * TEMP_HEIGHT * 2  # 18432 bytes
        self.rgb_size = WIDTH * HEIGHT * 3  # 172800 bytes
        self.frame_header_size = 24  # timestamp (8) + seq (4) + 3 floats (12)
        
        # Initialize file
        self._initialize()
    
    def _initialize(self):
        """Initialize file reading"""
        self._file_handle = open(self.file_path, "rb")
        
        # Read header: "TSEQ" + version (1 byte) + color flag (1 byte)
        header = self._file_handle.read(6)
        if len(header) != 6 or header[:4] != b"TSEQ":
            raise ValueError(f"Invalid file format: {self.file_path}")
        
        self._version = header[4]
        self._has_color = header[5] == 1
        
        # Calculate frame size
        frame_size = self.frame_header_size + self.yuyv_size + self.temp_size
        if self._has_color:
            frame_size += self.rgb_size
        
        # Calculate total frames
        file_size = self.file_path.stat().st_size
        header_size = 6
        self._total_frames = (file_size - header_size) // frame_size
        
        # Reset to beginning of frames
        self._file_handle.seek(6)
    
    def is_initialized(self) -> bool:
        """Check if reader is initialized"""
        return self._file_handle is not None
    
    def get_metadata(self) -> Optional[FrameMetadata]:
        """
        Get current frame metadata
        
        Returns:
            FrameMetadata named tuple, or None if no frame available
        """
        if self._current_metadata is None:
            # Try to read a frame first
            success, _ = self.read()
            if not success:
                return None
        
        # Return metadata in the same format as ThermalSharedMemory
        return FrameMetadata(
            seq=self._current_metadata['seq'],
            flag=1,  # Always 1 for recorded frames (already read)
            width=WIDTH,
            height=HEIGHT,
            min_temp=self._current_metadata['min_temp'],
            max_temp=self._current_metadata['max_temp'],
            avg_temp=self._current_metadata['avg_temp']
        )
    
    def has_new_frame(self) -> bool:
        """
        Check if a new frame is available
        
        Returns:
            True if there are more frames to read, False otherwise
        """
        if self._file_handle is None:
            return False
        
        # Check if we can read another frame
        current_pos = self._file_handle.tell()
        file_size = self.file_path.stat().st_size
        
        frame_size = self.frame_header_size + self.yuyv_size + self.temp_size
        if self._has_color:
            frame_size += self.rgb_size
        
        return (file_size - current_pos) >= frame_size
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame (VideoCapture-like interface)
        
        Returns:
            Tuple of (success, frame) where frame is BGR image (240x240x3)
        """
        if not self.has_new_frame():
            return False, None
        
        # Read frame header
        frame_header = self._file_handle.read(self.frame_header_size)
        if len(frame_header) < self.frame_header_size:
            return False, None
        
        # Unpack frame header
        timestamp, seq, min_temp, max_temp, avg_temp = struct.unpack("dIfff", frame_header)
        
        # Read YUYV data
        yuyv_bytes = self._file_handle.read(self.yuyv_size)
        if len(yuyv_bytes) < self.yuyv_size:
            return False, None
        yuyv = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
        
        # Read temperature array
        temp_bytes = self._file_handle.read(self.temp_size)
        if len(temp_bytes) < self.temp_size:
            return False, None
        temp_array = np.frombuffer(temp_bytes, dtype=np.uint16).reshape((TEMP_HEIGHT, TEMP_WIDTH))
        
        # Read RGB if present (skip it, we'll convert from YUYV)
        if self._has_color:
            rgb_bytes = self._file_handle.read(self.rgb_size)
            if len(rgb_bytes) < self.rgb_size:
                return False, None
        
        # Store current frame data
        self._current_frame_data = {
            'yuyv': yuyv,
            'temp_array': temp_array,
            'timestamp': timestamp
        }
        
        self._current_metadata = {
            'seq': seq,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'avg_temp': avg_temp
        }
        
        self._frame_index += 1
        
        # Convert YUYV to BGR for VideoCapture-like interface
        import cv2
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        
        return True, bgr
    
    def get_yuyv_frame(self) -> Optional[np.ndarray]:
        """
        Get current YUYV frame (ThermalSharedMemory interface)
        
        Returns:
            YUYV frame array (240x240x2), or None if no frame available
        """
        if self._current_frame_data is None:
            # Try to read a frame
            success, _ = self.read()
            if not success:
                return None
        
        return self._current_frame_data['yuyv'].copy()
    
    def get_temperature_array(self) -> Optional[np.ndarray]:
        """
        Get current temperature array (ThermalSharedMemory interface)
        
        Returns:
            Temperature array (96x96) with uint16 values, or None if no frame available
        """
        if self._current_frame_data is None:
            # Try to read a frame
            success, _ = self.read()
            if not success:
                return None
        
        return self._current_frame_data['temp_array'].copy()
    
    def mark_frame_read(self):
        """
        Mark current frame as read (ThermalSharedMemory interface)
        
        For recorded sequences, this clears the cached frame data so the next
        call to get_metadata()/get_yuyv_frame()/get_temperature_array() will
        read the next frame.
        """
        # Clear cached data so next iteration reads a new frame
        self._current_frame_data = None
        self._current_metadata = None
    
    def set(self, prop: int, value: float) -> bool:
        """
        Set property (VideoCapture-like interface)
        
        Args:
            prop: Property ID (e.g., cv2.CAP_PROP_POS_FRAMES)
            value: Property value
            
        Returns:
            True if successful, False otherwise
        """
        import cv2
        
        if prop == cv2.CAP_PROP_POS_FRAMES:
            # Seek to specific frame
            frame_index = int(value)
            if 0 <= frame_index < self._total_frames:
                frame_size = self.frame_header_size + self.yuyv_size + self.temp_size
                if self._has_color:
                    frame_size += self.rgb_size
                
                # Seek to frame position
                self._file_handle.seek(6 + frame_index * frame_size)
                self._frame_index = frame_index
                self._current_frame_data = None
                self._current_metadata = None
                return True
            return False
        
        elif prop == cv2.CAP_PROP_POS_MSEC:
            # Seek by time (approximate, based on frame rate)
            # This is approximate since we don't know the exact frame rate
            return False
        
        return False
    
    def get(self, prop: int) -> float:
        """
        Get property (VideoCapture-like interface)
        
        Args:
            prop: Property ID (e.g., cv2.CAP_PROP_POS_FRAMES, cv2.CAP_PROP_FRAME_COUNT)
            
        Returns:
            Property value
        """
        import cv2
        
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._frame_index)
        
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(WIDTH)
        
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(HEIGHT)
        
        elif prop == cv2.CAP_PROP_FPS:
            # Calculate approximate FPS from timestamps if available
            # For now, return 0 (unknown)
            return 0.0
        
        return 0.0
    
    def isOpened(self) -> bool:
        """
        Check if reader is opened (VideoCapture-like interface)
        
        Returns:
            True if file is open, False otherwise
        """
        return self._file_handle is not None
    
    def release(self):
        """
        Release resources (VideoCapture-like interface)
        """
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._current_frame_data = None
        self._current_metadata = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()

