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
    alpha_min: float = 0.4,
    alpha_max: float = 0.7,
    core_temp: float = 37.0,
    min_area: int = 50,
    temp_margin: float = 2.0,
    min_temp_above_env: float = 2.0,
    max_temp_limit: float = 42.0,
    max_aspect_ratio: float = 4.0,
    min_aspect_ratio: float = 0.4
) -> List[DetectedObject]:
    """
    Advanced human detection using adaptive temperature thresholds based on environment temperature.
    
    This method estimates the expected body temperature range from the environment temperature
    using the formula: Ts = Te + α × (Tc − Te)
    
    Where:
    - Ts = Skin temperature (estimated body temperature)
    - Te = Environment temperature
    - Tc = Core body temperature (default: 37°C)
    - α = Blood flow regulation coefficient (0.5-0.7 for face/torso)
    
    The detection focuses on warmer body parts (face/torso) and includes additional filters:
    - Minimum temperature above environment (to exclude cold objects)
    - Maximum temperature limit (to exclude hot objects like heaters)
    - Aspect ratio filtering (to match human body proportions)
    - Temperature consistency checks
    
    Args:
        temp_array: Temperature array (96x96, uint16 or float32)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        environment_temp: Environment/room temperature in Celsius. If None, will be estimated
                          from the frame using the 5th percentile method.
        alpha_min: Minimum alpha value for detection (default: 0.4, includes some cooler body parts)
        alpha_max: Maximum alpha value for detection (default: 0.7, face/torso)
        core_temp: Core body temperature in Celsius (default: 37.0)
        min_area: Minimum area in pixels for detected objects (default: 50)
        temp_margin: Temperature margin in Celsius to add around estimated range (default: 2.0)
        min_temp_above_env: Minimum temperature above environment to consider (default: 2.0°C)
                           This helps exclude objects that are only slightly warmer than room
        max_temp_limit: Maximum temperature limit to avoid detecting very hot objects (default: 42.0°C)
        max_aspect_ratio: Maximum aspect ratio (width/height or height/width) for human detection (default: 4.0)
        min_aspect_ratio: Minimum aspect ratio for human detection (default: 0.4)
    
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
    # Focus on face/torso (warmer parts) using alpha range 0.5-0.7
    from ..utils import estimate_body_temperature
    body_temp_min = estimate_body_temperature(env_temp, alpha=alpha_min, core_temp=core_temp)
    body_temp_max = estimate_body_temperature(env_temp, alpha=alpha_max, core_temp=core_temp)
    
    # Apply tighter bounds with margin
    detection_min = body_temp_min - temp_margin
    detection_max = body_temp_max + temp_margin
    
    # Ensure minimum temperature is significantly above environment (exclude cold objects)
    detection_min = max(detection_min, env_temp + min_temp_above_env)
    
    # Cap maximum temperature to avoid detecting very hot objects (heaters, lights, etc.)
    detection_max = min(detection_max, max_temp_limit)
    
    # Ensure we have a valid range
    if detection_min >= detection_max:
        return []
    
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
        
        # Filter by aspect ratio (humans have reasonable proportions)
        aspect_ratio_w_h = w / h if h > 0 else 0
        aspect_ratio_h_w = h / w if w > 0 else 0
        max_aspect = max(aspect_ratio_w_h, aspect_ratio_h_w)
        
        if max_aspect > max_aspect_ratio or max_aspect < min_aspect_ratio:
            continue
        
        # Calculate center
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        
        # Calculate temperature statistics for this object
        # Create a mask for this specific contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Extract temperatures within this contour
        object_temps = temp_celsius[contour_mask > 0]
        
        if len(object_temps) == 0:
            continue
        
        avg_temp = np.mean(object_temps)
        max_temp_obj = np.max(object_temps)
        min_temp_obj = np.min(object_temps)
        
        # Additional temperature consistency check
        # Human body parts should have relatively consistent temperatures
        temp_range = max_temp_obj - min_temp_obj
        temp_std = np.std(object_temps)
        
        # Filter out objects with too much temperature variation (likely not human)
        # Allow some variation but not excessive - relaxed slightly
        if temp_range > 10.0 or temp_std > 4.0:
            continue
        
        # Ensure average temperature is within reasonable human body range
        if avg_temp < detection_min or avg_temp > detection_max:
            continue
        
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

