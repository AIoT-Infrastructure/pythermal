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
from .detections import detect_object_centers, cluster_objects
from .detections.detector import DetectedObject

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalRecorder",
    "ThermalLiveView",
    "detect_object_centers",
    "cluster_objects",
    "DetectedObject",
    "WIDTH",
    "HEIGHT",
    "TEMP_WIDTH",
    "TEMP_HEIGHT",
]
