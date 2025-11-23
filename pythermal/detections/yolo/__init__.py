"""
YOLO detection module for thermal imaging.

Provides YOLO v11 implementations for:
- Object detection
- Pose detection
- Instance segmentation

Supports both default official models and custom thermal-specific models.
"""

from .object_detection import YOLOObjectDetector
from .pose_detection import YOLOPoseDetector

try:
    from .segmentation import YOLOSegmentationDetector
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    YOLOSegmentationDetector = None

__all__ = [
    "YOLOObjectDetector",
    "YOLOPoseDetector",
]

if SEGMENTATION_AVAILABLE:
    __all__.append("YOLOSegmentationDetector")

