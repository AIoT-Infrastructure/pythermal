#!/usr/bin/env python3
"""
Motion detection using background subtraction.

Provides background subtraction and motion detection for thermal images.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from collections import deque

from .utils import DetectedObject, convert_to_celsius


class BackgroundSubtractor:
    """
    Background subtraction for thermal images using running average.
    
    Maintains a background model and detects moving objects by comparing
    current frame to the background.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        history_size: int = 30,
        min_frames_for_background: int = 10
    ):
        """
        Initialize background subtractor.
        
        Args:
            learning_rate: Rate at which background model updates (0.0-1.0, default: 0.01)
                          Lower values = slower adaptation, more stable background
            history_size: Number of frames to keep in history for median-based background (default: 30)
            min_frames_for_background: Minimum frames needed before background is considered stable (default: 10)
        """
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.min_frames_for_background = min_frames_for_background
        
        self.background_model: Optional[np.ndarray] = None
        self.frame_history: deque = deque(maxlen=history_size)
        self.frame_count = 0
        self.background_ready = False
    
    def update(self, temp_celsius: np.ndarray) -> np.ndarray:
        """
        Update background model with new frame.
        
        Args:
            temp_celsius: Temperature array in Celsius (float32)
        
        Returns:
            Foreground mask (binary, uint8) - pixels that differ from background
        """
        if temp_celsius is None or temp_celsius.size == 0:
            return np.zeros((96, 96), dtype=np.uint8)
        
        # Ensure float32
        temp_celsius = temp_celsius.astype(np.float32)
        
        # Initialize background model on first frame
        if self.background_model is None:
            self.background_model = temp_celsius.copy()
            self.frame_count = 0
            self.background_ready = False
        
        # Add to history for median-based approach
        self.frame_history.append(temp_celsius.copy())
        self.frame_count += 1
        
        # Update background model using running average
        if self.frame_count < self.min_frames_for_background:
            # During initial frames, use simple average
            alpha = 1.0 / self.frame_count
            self.background_model = (1 - alpha) * self.background_model + alpha * temp_celsius
        else:
            # After minimum frames, use learning rate
            self.background_model = (
                (1 - self.learning_rate) * self.background_model +
                self.learning_rate * temp_celsius
            )
            self.background_ready = True
        
        # Calculate foreground mask (absolute difference)
        diff = np.abs(temp_celsius - self.background_model)
        
        return diff
    
    def get_background(self) -> Optional[np.ndarray]:
        """
        Get current background model.
        
        Returns:
            Background temperature array in Celsius, or None if not initialized
        """
        return self.background_model.copy() if self.background_model is not None else None
    
    def reset(self):
        """Reset background model."""
        self.background_model = None
        self.frame_history.clear()
        self.frame_count = 0
        self.background_ready = False
    
    def is_ready(self) -> bool:
        """
        Check if background model is ready for use.
        
        Returns:
            True if background has been initialized with enough frames
        """
        return self.background_ready


def detect_moving_objects(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float,
    background_subtractor: BackgroundSubtractor,
    temp_threshold: float = 2.0,
    min_area: int = 50,
    combine_with_temp_range: bool = True,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None
) -> Tuple[List[DetectedObject], np.ndarray]:
    """
    Detect moving objects using background subtraction.
    
    Args:
        temp_array: Temperature array (96x96, uint16 or float32)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        background_subtractor: BackgroundSubtractor instance
        temp_threshold: Temperature difference threshold in Celsius for motion detection (default: 2.0)
        min_area: Minimum area in pixels for detected objects (default: 50)
        combine_with_temp_range: If True, also filter by temperature range (default: True)
        temp_min: Minimum temperature for filtering (only used if combine_with_temp_range=True)
        temp_max: Maximum temperature for filtering (only used if combine_with_temp_range=True)
    
    Returns:
        Tuple of (list of DetectedObject instances, foreground mask)
    """
    if temp_array is None or temp_array.size == 0:
        return [], np.zeros((96, 96), dtype=np.uint8)
    
    # Convert to Celsius
    temp_celsius = convert_to_celsius(temp_array, min_temp, max_temp)
    
    # Update background and get foreground mask
    diff = background_subtractor.update(temp_celsius)
    
    # Create binary mask based on temperature difference
    motion_mask = (diff >= temp_threshold).astype(np.uint8) * 255
    
    # Optionally combine with temperature range filtering
    if combine_with_temp_range:
        if temp_min is not None and temp_max is not None:
            temp_mask = ((temp_celsius >= temp_min) & (temp_celsius <= temp_max)).astype(np.uint8) * 255
            # Combine both masks (object must be moving AND in temperature range)
            motion_mask = cv2.bitwise_and(motion_mask, temp_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        # Calculate temperature statistics for this object
        contour_mask = np.zeros_like(motion_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Extract temperatures within this contour
        object_temps = temp_celsius[contour_mask > 0]
        
        if len(object_temps) > 0:
            avg_temp = np.mean(object_temps)
            max_temp_obj = np.max(object_temps)
            min_temp_obj = np.min(object_temps)
        else:
            # Fallback: use center pixel temperature
            cy_int = int(np.clip(center_y, 0, temp_celsius.shape[0] - 1))
            cx_int = int(np.clip(center_x, 0, temp_celsius.shape[1] - 1))
            avg_temp = temp_celsius[cy_int, cx_int]
            max_temp_obj = avg_temp
            min_temp_obj = avg_temp
        
        detected_objects.append(DetectedObject(
            center_x=center_x,
            center_y=center_y,
            width=w,
            height=h,
            area=int(area),
            avg_temperature=float(avg_temp),
            max_temperature=float(max_temp_obj),
            min_temperature=float(min_temp_obj)
        ))
    
    return detected_objects, motion_mask

