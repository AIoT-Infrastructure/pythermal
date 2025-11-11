"""
PyThermal - A lightweight Python library for thermal sensing and analytics.

A lightweight Python library for thermal sensing and analytics on ARM Linux platforms.
"""

__version__ = "0.1.0"

from .device import ThermalDevice
from .thermal_shared_memory import ThermalSharedMemory, FrameMetadata
from .record import ThermalRecorder
from .live_view import ThermalLiveView

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalRecorder",
    "ThermalLiveView",
]
