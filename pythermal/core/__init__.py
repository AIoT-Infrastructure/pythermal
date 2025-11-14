"""
Core thermal camera components

This module contains the core components for thermal camera access:
- ThermalDevice: Device management
- ThermalSharedMemory: Shared memory interface
- ThermalSequenceReader: Reader for recorded sequences
- ThermalCapture: Unified capture interface (live or recorded)
"""

from .device import ThermalDevice
from .thermal_shared_memory import (
    ThermalSharedMemory,
    FrameMetadata,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
    SHM_NAME,
    FRAME_SZ,
    TEMP_DATA_SIZE,
)
from .sequence_reader import ThermalSequenceReader
from .capture import ThermalCapture

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalSequenceReader",
    "ThermalCapture",
    "WIDTH",
    "HEIGHT",
    "TEMP_WIDTH",
    "TEMP_HEIGHT",
    "SHM_NAME",
    "FRAME_SZ",
    "TEMP_DATA_SIZE",
]

