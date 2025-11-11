"""
Detection module for thermal object detection and clustering.

Provides functions to detect object centers based on temperature ranges
and cluster detected objects for visualization.
"""

from .detector import detect_object_centers, cluster_objects

__all__ = [
    "detect_object_centers",
    "cluster_objects",
]

