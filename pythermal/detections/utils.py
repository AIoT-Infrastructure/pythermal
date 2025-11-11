#!/usr/bin/env python3
"""
Shared utilities for thermal object detection.

Provides common data structures and utility functions used across
different detection modules.
"""

import numpy as np
from typing import List
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

