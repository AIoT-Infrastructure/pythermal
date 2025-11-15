#!/usr/bin/env python3
"""
Temperature-based object detection.

Provides functions to detect objects based on temperature ranges.
"""

import numpy as np
import cv2
from typing import List, Optional

from .utils import DetectedObject, convert_to_celsius
from ..utils import (
    estimate_environment_temperature_v1,
    estimate_body_temperature_range,
)


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


def detect_humans_adaptive(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float,
    environment_temp: Optional[float] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 0.7,
    core_temp: float = 37.0,
    min_area: int = 50,
    temp_margin: float = 2.0
) -> List[DetectedObject]:
    """
    Advanced human detection using adaptive temperature thresholds based on environment temperature.
    
    This method estimates the expected body temperature range from the environment temperature
    using the formula: Ts = Te + α × (Tc − Te)
    
    Where:
    - Ts = Skin temperature (estimated body temperature)
    - Te = Environment temperature
    - Tc = Core body temperature (default: 37°C)
    - α = Blood flow regulation coefficient (0.2-0.7)
    
    The detection uses adaptive thresholds that account for:
    - Face/torso: α ≈ 0.5–0.7 (warmer body parts)
    - Hands/feet: α ≈ 0.2–0.4 (cooler body parts)
    
    Args:
        temp_array: Temperature array (96x96, uint16 or float32)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        environment_temp: Environment/room temperature in Celsius. If None, will be estimated
                          from the frame using the 5th percentile method.
        alpha_min: Minimum alpha value for hands/feet (default: 0.2)
        alpha_max: Maximum alpha value for face/torso (default: 0.7)
        core_temp: Core body temperature in Celsius (default: 37.0)
        min_area: Minimum area in pixels for detected objects (default: 50)
        temp_margin: Temperature margin in Celsius to add above estimated range (default: 2.0)
                    This accounts for measurement variations and ensures detection
    
    Returns:
        List of DetectedObject instances representing detected humans
    
    Examples:
        >>> # Detect humans with estimated environment temperature
        >>> objects = detect_humans_adaptive(temp_array, min_temp, max_temp)
        
        >>> # Detect humans with known room temperature
        >>> objects = detect_humans_adaptive(temp_array, min_temp, max_temp, environment_temp=22.0)
    """
    if temp_array is None or temp_array.size == 0:
        return []
    
    # Estimate environment temperature if not provided
    if environment_temp is None:
        env_temp = estimate_environment_temperature_v1(temp_array, min_temp, max_temp)
        if env_temp is None:
            return []
    else:
        env_temp = environment_temp
    
    # Convert to Celsius
    temp_celsius = convert_to_celsius(temp_array, min_temp, max_temp)
    
    # Estimate body temperature range based on environment
    # This gives us the expected temperature range for human body parts
    body_temp_min, body_temp_max = estimate_body_temperature_range(
        env_temp, core_temp=core_temp
    )
    
    # Use the minimum alpha (hands/feet) as lower bound and maximum alpha (face/torso) as upper bound
    # Add a margin to account for measurement variations and ensure we don't miss detections
    detection_min = body_temp_min - temp_margin
    detection_max = body_temp_max + temp_margin
    
    # Also ensure we don't go below environment temperature (no negative detection)
    detection_min = max(detection_min, env_temp)
    
    # Create binary mask for adaptive temperature range
    mask = ((temp_celsius >= detection_min) & (temp_celsius <= detection_max)).astype(np.uint8) * 255
    
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

