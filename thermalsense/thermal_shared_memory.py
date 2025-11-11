#!/usr/bin/env python3
"""
Thermal Shared Memory Reader

Provides a reusable class for reading thermal camera data from shared memory.
Handles YUYV frame data, temperature arrays, and metadata.
"""

import mmap
import struct
import numpy as np
import os
from typing import Optional, NamedTuple


# Constants
SHM_NAME = "/dev/shm/yuyv240_shm"
WIDTH = 240
HEIGHT = 240
FRAME_SZ = WIDTH * HEIGHT * 2
TEMP_WIDTH = 96
TEMP_HEIGHT = 96
TEMP_DATA_SIZE = TEMP_WIDTH * TEMP_HEIGHT * 2  # 16-bit integers

# Shared memory layout:
# data[0:FRAME_SZ] = YUYV data
# temp_data[FRAME_SZ:FRAME_SZ+TEMP_DATA_SIZE] = temperature data (96x96, 16-bit)
# Metadata from end: seq[-32:-28], flag[-28:-24], width[-24:-20], height[-20:-16], temps[-16:-4], reserved[-4:]


class FrameMetadata(NamedTuple):
    """Frame metadata structure"""
    seq: int
    flag: int
    width: int
    height: int
    min_temp: float
    max_temp: float
    avg_temp: float


class ThermalSharedMemory:
    """
    Reader for thermal camera shared memory
    
    Provides methods to read YUYV frames, temperature arrays, and metadata
    from the shared memory buffer created by the C++ thermal capture system.
    """
    
    def __init__(self, shm_name: str = SHM_NAME):
        """
        Initialize thermal shared memory reader
        
        Args:
            shm_name: Path to shared memory file (default: /dev/shm/yuyv240_shm)
        """
        self.shm_name = shm_name
        self.shm = None
        self._file_handle = None
        
    def initialize(self) -> bool:
        """
        Initialize shared memory connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.shm_name):
                return False
                
            self._file_handle = open(self.shm_name, "r+b")
            # Size: FRAME_SZ + TEMP_DATA_SIZE + 32 bytes (metadata)
            total_size = FRAME_SZ + TEMP_DATA_SIZE + 32
            self.shm = mmap.mmap(self._file_handle.fileno(), total_size)
            
            return True
            
        except Exception as e:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            self.shm = None
            return False
    
    def is_initialized(self) -> bool:
        """Check if shared memory is initialized"""
        return self.shm is not None
    
    def get_metadata(self) -> Optional[FrameMetadata]:
        """
        Read frame metadata from shared memory
        
        Returns:
            FrameMetadata named tuple, or None if not initialized
        """
        if not self.shm:
            return None
        
        try:
            seq = struct.unpack("I", self.shm[-32:-28])[0]
            flag = struct.unpack("I", self.shm[-28:-24])[0]
            width = struct.unpack("I", self.shm[-24:-20])[0]
            height = struct.unpack("I", self.shm[-20:-16])[0]
            min_temp, max_temp, avg_temp = struct.unpack("fff", self.shm[-16:-4])
            
            return FrameMetadata(
                seq=seq,
                flag=flag,
                width=width,
                height=height,
                min_temp=min_temp,
                max_temp=max_temp,
                avg_temp=avg_temp
            )
        except Exception:
            return None
    
    def has_new_frame(self) -> bool:
        """
        Check if a new frame is available
        
        Returns:
            True if flag is set (new frame available), False otherwise
        """
        metadata = self.get_metadata()
        return metadata is not None and metadata.flag == 1
    
    def get_yuyv_frame(self) -> Optional[np.ndarray]:
        """
        Read YUYV frame data from shared memory
        
        Returns:
            numpy array of shape (HEIGHT, WIDTH, 2) with YUYV data, or None if not initialized
        """
        if not self.shm:
            return None
        
        try:
            yuyv_bytes = self.shm[:FRAME_SZ]
            yuyv = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
            return yuyv
        except Exception:
            return None
    
    def get_temperature_array(self) -> Optional[np.ndarray]:
        """
        Read raw temperature array (96x96) from shared memory
        
        Returns:
            numpy array of shape (TEMP_HEIGHT, TEMP_WIDTH) with uint16 temperature values,
            or None if not initialized
        """
        if not self.shm:
            return None
        
        try:
            # Temperature data is stored as 16-bit integers starting at FRAME_SZ
            temp_bytes = self.shm[FRAME_SZ:FRAME_SZ + TEMP_DATA_SIZE]
            # Convert to numpy array of uint16, then reshape to 96x96
            temp_array = np.frombuffer(temp_bytes, dtype=np.uint16).reshape((TEMP_HEIGHT, TEMP_WIDTH))
            return temp_array
        except Exception:
            return None
    
    def get_temperature_map_celsius(self) -> Optional[np.ndarray]:
        """
        Get temperature map in Celsius (96x96)
        
        Converts the raw temperature array to actual Celsius values using
        the metadata min/max temperatures for calibration.
        
        Returns:
            numpy array of shape (TEMP_HEIGHT, TEMP_WIDTH) with float32 Celsius values,
            or None if not initialized or metadata unavailable
        """
        if not self.shm:
            return None
        
        # Get raw temperature array
        raw_array = self.get_temperature_array()
        if raw_array is None:
            return None
        
        # Get metadata for temperature range
        metadata = self.get_metadata()
        if metadata is None:
            return None
        
        min_temp = metadata.min_temp
        max_temp = metadata.max_temp
        
        # Convert to float32 for processing
        temp_float = raw_array.astype(np.float32)
        
        # Get actual min/max from the raw array
        raw_min = np.min(temp_float)
        raw_max = np.max(temp_float)
        raw_range = raw_max - raw_min
        
        # Map raw values to Celsius temperature range
        if raw_range > 0:
            # Normalize raw values to 0-1 range, then map to temperature range
            normalized = (temp_float - raw_min) / raw_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
            return temp_celsius
        else:
            # All values are the same, return array filled with average temperature
            avg_temp = metadata.avg_temp
            return np.full((TEMP_HEIGHT, TEMP_WIDTH), avg_temp, dtype=np.float32)
    
    def mark_frame_read(self) -> bool:
        """
        Mark the current frame as read by setting flag to 0
        
        This should be called after reading frame data to allow the producer
        to write new frames.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.shm:
            return False
        
        try:
            self.shm[-28:-24] = struct.pack("I", 0)
            return True
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up resources and close shared memory"""
        if self.shm:
            self.shm.close()
            self.shm = None
        
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

