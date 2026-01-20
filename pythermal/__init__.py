"""
PyThermal - A lightweight Python library for thermal sensing and analytics.

A lightweight Python library for thermal sensing and analytics on ARM Linux platforms.
"""

__version__ = "0.2.11"

from .core import (
    ThermalDevice,
    ThermalSharedMemory,
    FrameMetadata,
    ThermalSequenceReader,
    ThermalCapture,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from .record import ThermalRecorder
from .live_view import ThermalLiveView
from .detections import (
    detect_object_centers,
    detect_humans_adaptive,
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

# Optional synthesis module (requires YOLO dependencies)
try:
    from .synthesis import (
        SyntheticThermalGenerator,
        ImageProcessor,
        HumanSegmenter,
        TemperatureMapper,
        ThermalFrameGenerator,
    )
    SYNTHESIS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AVAILABLE = False
    SyntheticThermalGenerator = None
    ImageProcessor = None
    HumanSegmenter = None
    TemperatureMapper = None
    ThermalFrameGenerator = None

__all__ = [
    "ThermalDevice",
    "ThermalSharedMemory",
    "FrameMetadata",
    "ThermalRecorder",
    "ThermalLiveView",
    "ThermalSequenceReader",
    "ThermalCapture",
    "detect_object_centers",
    "detect_humans_adaptive",
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

# Add synthesis exports if available
if SYNTHESIS_AVAILABLE:
    __all__.extend([
        "SyntheticThermalGenerator",
        "ImageProcessor",
        "HumanSegmenter",
        "TemperatureMapper",
        "ThermalFrameGenerator",
    ])
