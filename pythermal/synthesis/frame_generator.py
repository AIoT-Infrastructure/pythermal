#!/usr/bin/env python3
"""
Thermal Frame Generator for Synthetic Data

Generates ThermalFrame objects from processed RGB images and temperature maps.
Converts temperature maps to the required formats (YUYV, temperature array).
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple
from ..core.frame_processor import ThermalFrame, FrameMetadata
from ..core.thermal_shared_memory import WIDTH, HEIGHT, TEMP_WIDTH, TEMP_HEIGHT


class ThermalFrameGenerator:
    """
    Generates ThermalFrame objects from RGB images and temperature maps.
    
    Converts:
    - RGB image -> YUYV format
    - Temperature map -> Temperature array (96x96, uint16)
    - Creates metadata with temperature statistics
    """
    
    def __init__(self, temp_min: float = 20.0, temp_max: float = 40.0):
        """
        Initialize frame generator.
        
        Args:
            temp_min: Minimum temperature for uint16 mapping (Celsius)
            temp_max: Maximum temperature for uint16 mapping (Celsius)
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
    
    def generate_frame(
        self,
        rgb_image: np.ndarray,
        temp_map_96: np.ndarray,
        timestamp: Optional[float] = None,
        sequence: int = 0,
    ) -> ThermalFrame:
        """
        Generate ThermalFrame from RGB image and temperature map.
        
        Args:
            rgb_image: RGB image (240, 240, 3), uint8
            temp_map_96: Temperature map (96, 96), float32, Celsius
            timestamp: Optional timestamp (defaults to current time)
            sequence: Frame sequence number
        
        Returns:
            ThermalFrame object
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Validate inputs
        if rgb_image.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"RGB image must be ({HEIGHT}, {WIDTH}, 3), got {rgb_image.shape}")
        
        if temp_map_96.shape != (TEMP_HEIGHT, TEMP_WIDTH):
            raise ValueError(f"Temperature map must be ({TEMP_HEIGHT}, {TEMP_WIDTH}), got {temp_map_96.shape}")
        
        # Convert RGB to YUYV
        yuyv = self._rgb_to_yuyv(rgb_image)
        
        # Convert temperature map to uint16 array
        temp_array = self._temp_map_to_array(temp_map_96)
        
        # Calculate metadata
        metadata = self._calculate_metadata(temp_map_96, sequence)
        
        return ThermalFrame(
            timestamp=timestamp,
            metadata=metadata,
            yuyv=yuyv,
            temp_array=temp_array,
            rgb=rgb_image.copy(),
        )
    
    def _rgb_to_yuyv(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to YUYV format.
        
        Args:
            rgb_image: RGB image (240, 240, 3), uint8
        
        Returns:
            YUYV array (240, 240, 2), uint8
        """
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Convert BGR to YUV (full resolution)
        yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
        
        h, w = yuv_image.shape[:2]
        yuyv = np.zeros((h, w, 2), dtype=np.uint8)
        
        y_channel = yuv_image[:, :, 0]
        u_channel = yuv_image[:, :, 1]
        v_channel = yuv_image[:, :, 2]
        
        # YUYV format (4:2:2): Y0 U Y1 V Y2 U Y3 V ...
        # We store as: [Y, chroma] where chroma alternates U/V
        # Y channel is full resolution
        yuyv[:, :, 0] = y_channel
        
        # U and V are subsampled horizontally (every other pixel)
        # Interleave U and V: U for even columns, V for odd columns
        yuyv[:, ::2, 1] = u_channel[:, ::2]  # U (subsampled to even columns)
        yuyv[:, 1::2, 1] = v_channel[:, 1::2]  # V (subsampled to odd columns)
        
        return yuyv
    
    def _temp_map_to_array(self, temp_map: np.ndarray) -> np.ndarray:
        """
        Convert temperature map (float32, Celsius) to uint16 array.
        
        Maps temperature range [temp_min, temp_max] to [0, 65535].
        
        Args:
            temp_map: Temperature map (96, 96), float32, Celsius
        
        Returns:
            Temperature array (96, 96), uint16
        """
        # Clamp temperatures to valid range
        clamped = np.clip(temp_map, self.temp_min, self.temp_max)
        
        # Normalize to [0, 1]
        temp_range = self.temp_max - self.temp_min
        if temp_range > 0:
            normalized = (clamped - self.temp_min) / temp_range
        else:
            normalized = np.zeros_like(clamped)
        
        # Map to [0, 65535]
        uint16_array = (normalized * 65535.0).astype(np.uint16)
        
        return uint16_array
    
    def _calculate_metadata(self, temp_map: np.ndarray, sequence: int) -> FrameMetadata:
        """
        Calculate frame metadata from temperature map.
        
        Args:
            temp_map: Temperature map (96, 96), float32, Celsius
            sequence: Frame sequence number
        
        Returns:
            FrameMetadata object
        """
        min_temp = float(np.min(temp_map))
        max_temp = float(np.max(temp_map))
        avg_temp = float(np.mean(temp_map))
        
        return FrameMetadata(
            seq=sequence,
            flag=1,
            width=WIDTH,
            height=HEIGHT,
            min_temp=min_temp,
            max_temp=max_temp,
            avg_temp=avg_temp,
        )

