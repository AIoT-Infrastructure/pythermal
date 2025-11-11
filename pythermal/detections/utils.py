#!/usr/bin/env python3
"""
Shared utilities for thermal object detection.

Provides common data structures and utility functions used across
different detection modules.
"""

import numpy as np
import cv2
from typing import List, Optional, Callable
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


def convert_to_celsius(
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
    if temp_array is None or temp_array.size == 0:
        return np.array([])
    
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


def calculate_aspect_ratio(obj: DetectedObject) -> float:
    """
    Calculate aspect ratio of detected object.
    
    Args:
        obj: DetectedObject instance
    
    Returns:
        Aspect ratio (width/height). Values > 1.0 indicate wider objects.
    """
    if obj.height == 0:
        return float('inf')
    return obj.width / obj.height


def calculate_compactness(obj: DetectedObject) -> float:
    """
    Calculate compactness (circularity approximation) of detected object.
    
    Uses bounding box approximation: 4π*area/(width+height)²
    Higher values (closer to 1.0) indicate more circular/compact objects.
    
    Args:
        obj: DetectedObject instance
    
    Returns:
        Compactness value (0.0 to 1.0)
    """
    if obj.width == 0 and obj.height == 0:
        return 0.0
    perimeter_approx = 2 * (obj.width + obj.height)
    if perimeter_approx == 0:
        return 0.0
    return (4 * np.pi * obj.area) / (perimeter_approx ** 2)


def calculate_circularity(contour: np.ndarray, area: float) -> float:
    """
    Calculate true circularity from contour.
    
    Circularity = 4π*area/perimeter²
    Higher values (closer to 1.0) indicate more circular objects.
    
    Args:
        contour: Contour points (numpy array)
        area: Contour area
    
    Returns:
        Circularity value (0.0 to 1.0)
    """
    if area == 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def calculate_convexity_ratio(contour: np.ndarray, area: float) -> float:
    """
    Calculate convexity ratio from contour.
    
    Convexity = area / convex_hull_area
    Higher values (closer to 1.0) indicate more convex objects.
    
    Args:
        contour: Contour points (numpy array)
        area: Contour area
    
    Returns:
        Convexity ratio (0.0 to 1.0)
    """
    if area == 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def filter_by_aspect_ratio(
    objects: List[DetectedObject],
    min_ratio: Optional[float] = None,
    max_ratio: Optional[float] = None
) -> List[DetectedObject]:
    """
    Filter objects by aspect ratio (width/height).
    
    Args:
        objects: List of DetectedObject instances
        min_ratio: Minimum aspect ratio (default: None, no minimum)
        max_ratio: Maximum aspect ratio (default: None, no maximum)
    
    Returns:
        Filtered list of objects
    """
    filtered = []
    for obj in objects:
        ratio = calculate_aspect_ratio(obj)
        if min_ratio is not None and ratio < min_ratio:
            continue
        if max_ratio is not None and ratio > max_ratio:
            continue
        filtered.append(obj)
    return filtered


def filter_by_compactness(
    objects: List[DetectedObject],
    min_compactness: Optional[float] = None,
    max_compactness: Optional[float] = None
) -> List[DetectedObject]:
    """
    Filter objects by compactness (circularity approximation).
    
    Args:
        objects: List of DetectedObject instances
        min_compactness: Minimum compactness (default: None, no minimum)
        max_compactness: Maximum compactness (default: None, no maximum)
    
    Returns:
        Filtered list of objects
    """
    filtered = []
    for obj in objects:
        compactness = calculate_compactness(obj)
        if min_compactness is not None and compactness < min_compactness:
            continue
        if max_compactness is not None and compactness > max_compactness:
            continue
        filtered.append(obj)
    return filtered


def filter_by_area(
    objects: List[DetectedObject],
    min_area: Optional[int] = None,
    max_area: Optional[int] = None
) -> List[DetectedObject]:
    """
    Filter objects by area.
    
    Args:
        objects: List of DetectedObject instances
        min_area: Minimum area in pixels (default: None, no minimum)
        max_area: Maximum area in pixels (default: None, no maximum)
    
    Returns:
        Filtered list of objects
    """
    filtered = []
    for obj in objects:
        if min_area is not None and obj.area < min_area:
            continue
        if max_area is not None and obj.area > max_area:
            continue
        filtered.append(obj)
    return filtered


def filter_by_shape(
    objects: List[DetectedObject],
    min_aspect_ratio: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
    min_compactness: Optional[float] = None,
    max_compactness: Optional[float] = None,
    min_area: Optional[int] = None,
    max_area: Optional[int] = None
) -> List[DetectedObject]:
    """
    Filter objects by multiple shape criteria.
    
    Convenience function that applies all shape filters at once.
    
    Args:
        objects: List of DetectedObject instances
        min_aspect_ratio: Minimum aspect ratio (width/height)
        max_aspect_ratio: Maximum aspect ratio (width/height)
        min_compactness: Minimum compactness (0.0-1.0)
        max_compactness: Maximum compactness (0.0-1.0)
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels
    
    Returns:
        Filtered list of objects
    """
    filtered = objects
    
    if min_aspect_ratio is not None or max_aspect_ratio is not None:
        filtered = filter_by_aspect_ratio(filtered, min_aspect_ratio, max_aspect_ratio)
    
    if min_compactness is not None or max_compactness is not None:
        filtered = filter_by_compactness(filtered, min_compactness, max_compactness)
    
    if min_area is not None or max_area is not None:
        filtered = filter_by_area(filtered, min_area, max_area)
    
    return filtered

