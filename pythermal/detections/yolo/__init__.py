"""
YOLO detection module for thermal imaging.

Provides YOLO v11 implementations for:
- Object detection
- Pose detection

Supports both default official models and custom thermal-specific models.
"""

from .object_detection import YOLOObjectDetector
from .pose_detection import YOLOPoseDetector

__all__ = [
    "YOLOObjectDetector",
    "YOLOPoseDetector",
]

