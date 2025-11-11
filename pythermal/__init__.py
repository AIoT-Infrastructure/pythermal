"""
PyThermal - A lightweight Python library for thermal sensing and analytics.

A lightweight Python library for thermal sensing and analytics on ARM Linux platforms.
"""

__version__ = "0.1.0"

from .device import ThermalDevice
from .thermal_shared_memory import (
    ThermalSharedMemory,
    FrameMetadata,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from .record import ThermalRecorder
from .live_view import ThermalLiveView
from .detections import (
    detect_object_centers,
    cluster_objects,
    BackgroundSubtractor,
    detect_moving_objects,
    ROI,
    ROIManager,
    DetectedObject,
    calculate_aspect_ratio,
    calculate_compactness,
    calculate_circularity,
    calculate_convexity_ratio,
    filter_by_aspect_ratio,
    filter_by_compactness,
    filter_by_area,
    filter_by_shape,
)

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalRecorder",
    "ThermalLiveView",
    "detect_object_centers",
    "cluster_objects",
    "BackgroundSubtractor",
    "detect_moving_objects",
    "ROI",
    "ROIManager",
    "DetectedObject",
    "calculate_aspect_ratio",
    "calculate_compactness",
    "calculate_circularity",
    "calculate_convexity_ratio",
    "filter_by_aspect_ratio",
    "filter_by_compactness",
    "filter_by_area",
    "filter_by_shape",
    "WIDTH",
    "HEIGHT",
    "TEMP_WIDTH",
    "TEMP_HEIGHT",
]
