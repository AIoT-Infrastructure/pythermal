"""
Detection module for thermal object detection and clustering.

Provides functions for:
- Temperature-based object detection
- Motion detection using background subtraction
- Region of Interest (ROI) management and zone monitoring
- YOLO v11 object and pose detection
"""

from .utils import (
    DetectedObject,
    convert_to_celsius,
    cluster_objects,
    calculate_aspect_ratio,
    calculate_compactness,
    calculate_circularity,
    calculate_convexity_ratio,
    filter_by_aspect_ratio,
    filter_by_compactness,
    filter_by_area,
    filter_by_shape,
)
from .temperature_detection import detect_object_centers, detect_humans_adaptive
from .motion_detection import BackgroundSubtractor, detect_moving_objects
from .roi import ROI, ROIManager

# YOLO detection modules (optional, requires ultralytics)
try:
    from .yolo import YOLOObjectDetector, YOLOPoseDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOObjectDetector = None
    YOLOPoseDetector = None

__all__ = [
    # Utilities
    "DetectedObject",
    "convert_to_celsius",
    "cluster_objects",
    # Shape analysis
    "calculate_aspect_ratio",
    "calculate_compactness",
    "calculate_circularity",
    "calculate_convexity_ratio",
    "filter_by_aspect_ratio",
    "filter_by_compactness",
    "filter_by_area",
    "filter_by_shape",
    # Temperature detection
    "detect_object_centers",
    "detect_humans_adaptive",
    # Motion detection
    "BackgroundSubtractor",
    "detect_moving_objects",
    # ROI management
    "ROI",
    "ROIManager",
    # YOLO detection (optional)
    "YOLOObjectDetector",
    "YOLOPoseDetector",
]
