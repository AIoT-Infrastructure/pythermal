#!/usr/bin/env python3
"""
Temperature-based object detection.

Provides functions to detect objects based on temperature ranges.
"""

import numpy as np
import cv2
from typing import List

from .utils import DetectedObject, convert_to_celsius


def detect_object_centers(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float,
    temp_min: float = 31.0,
    temp_max: float = 39.0,
    min_area: int = 50
) -> List[DetectedObject]:
    """
    Detect object centers from temperature map based on temperature range.
    
    Args:
        temp_array: Temperature array (96x96, uint16 or float32)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        temp_min: Minimum temperature threshold in Celsius (default: 31.0 for human body)
        temp_max: Maximum temperature threshold in Celsius (default: 39.0 for human body)
        min_area: Minimum area in pixels for detected objects (default: 50)
    
    Returns:
        List of DetectedObject instances with center coordinates and properties
    """
    if temp_array is None or temp_array.size == 0:
        return []
    
    # Convert to Celsius
    temp_celsius = convert_to_celsius(temp_array, min_temp, max_temp)
    
    # Create binary mask for temperature range
    mask = ((temp_celsius >= temp_min) & (temp_celsius <= temp_max)).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        # Create a mask for this specific contour
        contour_mask = np.zeros_like(mask)
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
    
    return detected_objects

