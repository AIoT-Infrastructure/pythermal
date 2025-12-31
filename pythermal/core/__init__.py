"""
Core thermal camera components

This module contains the core components for thermal camera access:
- ThermalDevice: Device management
- ThermalSharedMemory: Shared memory interface
- ThermalSequenceReader: Reader for recorded sequences
- ThermalCapture: Unified capture interface (live or recorded)
- ThermalFrameProcessor: Process and replay individual frames
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
    get_shm_name,
)
from .sequence_reader import ThermalSequenceReader
from .capture import ThermalCapture
from .frame_processor import ThermalFrameProcessor, ThermalFrame
from .device_manager import DeviceManager, get_device_id_by_serial, get_smallest_device_id

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalSequenceReader",
    "ThermalCapture",
    "ThermalFrameProcessor",
    "ThermalFrame",
    "DeviceManager",
    "get_device_id_by_serial",
    "get_smallest_device_id",
    "WIDTH",
    "HEIGHT",
    "TEMP_WIDTH",
    "TEMP_HEIGHT",
    "SHM_NAME",
    "FRAME_SZ",
    "TEMP_DATA_SIZE",
    "get_shm_name",
]

