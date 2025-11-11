"""
Detection module for thermal object detection and clustering.

Provides functions for:
- Temperature-based object detection
- Motion detection using background subtraction
- Region of Interest (ROI) management and zone monitoring
"""

from .utils import DetectedObject, convert_to_celsius, cluster_objects
from .temperature_detection import detect_object_centers
from .motion_detection import BackgroundSubtractor, detect_moving_objects
from .roi import ROI, ROIManager

__all__ = [
    # Utilities
    "DetectedObject",
    "convert_to_celsius",
    "cluster_objects",
    # Temperature detection
    "detect_object_centers",
    # Motion detection
    "BackgroundSubtractor",
    "detect_moving_objects",
    # ROI management
    "ROI",
    "ROIManager",
]
