#!/usr/bin/env python3
"""
Thermal Object Detection Module

Provides functions to detect object centers based on temperature ranges.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents a detected object with its center and properties"""
    center_x: float
    center_y: float
    width: int
    height: int
    area: int
    avg_temperature: float
    max_temperature: float
    min_temperature: float


def _convert_to_celsius(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float
) -> np.ndarray:
    """
    Convert temperature array to Celsius.
    
    Args:
        temp_array: Temperature array (uint16 or float32)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
    
    Returns:
        Temperature array in Celsius (float32)
    """
    if temp_array.dtype == np.uint16:
        # Convert uint16 to Celsius using min/max from metadata
        raw_min = np.min(temp_array)
        raw_max = np.max(temp_array)
        raw_range = raw_max - raw_min
        
        if raw_range > 0:
            # Normalize raw values to 0-1 range, then map to temperature range
            normalized = (temp_array.astype(np.float32) - raw_min) / raw_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
        else:
            # All values are the same
            temp_celsius = np.full_like(temp_array, (min_temp + max_temp) / 2.0, dtype=np.float32)
    else:
        # Already in Celsius
        temp_celsius = temp_array.astype(np.float32)
    
    return temp_celsius


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
    temp_celsius = _convert_to_celsius(temp_array, min_temp, max_temp)
    
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


def cluster_objects(
    objects: List[DetectedObject],
    max_distance: float = 30.0
) -> List[List[DetectedObject]]:
    """
    Cluster detected objects that are close to each other.
    
    Uses simple distance-based clustering.
    
    Args:
        objects: List of DetectedObject instances
        max_distance: Maximum distance between objects to be in the same cluster (default: 30.0)
    
    Returns:
        List of clusters, where each cluster is a list of DetectedObject instances
    """
    if not objects:
        return []
    
    clusters = []
    used = set()
    
    for i, obj in enumerate(objects):
        if i in used:
            continue
        
        # Start a new cluster with this object
        cluster = [obj]
        used.add(i)
        
        # Find all objects within max_distance
        changed = True
        while changed:
            changed = False
            for j, other_obj in enumerate(objects):
                if j in used:
                    continue
                
                # Check distance to any object in current cluster
                for cluster_obj in cluster:
                    distance = np.sqrt(
                        (cluster_obj.center_x - other_obj.center_x) ** 2 +
                        (cluster_obj.center_y - other_obj.center_y) ** 2
                    )
                    if distance <= max_distance:
                        cluster.append(other_obj)
                        used.add(j)
                        changed = True
                        break
        
        clusters.append(cluster)
    
    return clusters

