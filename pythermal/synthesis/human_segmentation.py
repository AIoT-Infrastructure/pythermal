#!/usr/bin/env python3
"""
Human Segmentation for Synthetic Thermal Generation

Uses YOLO segmentation to segment humans from RGB images.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from ..detections.yolo.segmentation import YOLOSegmentationDetector


class HumanSegmenter:
    """
    Segments humans from RGB images using YOLO segmentation.
    
    Uses YOLO segmentation models to create precise pixel-level masks.
    """
    
    # COCO class ID for "person"
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_size: str = "nano",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize human segmenter.
        
        Args:
            model_size: YOLO segmentation model size ("nano", "small", "medium", "large", "xlarge")
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu", "cuda", etc.)
        """
        self.detector = YOLOSegmentationDetector(
            model_size=model_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segment humans from image using segmentation masks.
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            Tuple of:
            - mask: Binary mask (H, W) where 255 indicates human pixels
            - detections: List of segmentation detection dictionaries with masks
        """
        # Convert RGB to BGR for YOLO (OpenCV format)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect and segment persons
        detections = self.detector.detect(bgr_image, classes=[self.PERSON_CLASS_ID])
        
        # Combine all segmentation masks into a single mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for det in detections:
            seg_mask = det.get("mask")
            if seg_mask is not None:
                # Combine masks (use maximum to handle overlaps)
                mask = np.maximum(mask, seg_mask)
        
        return mask, detections
    
    def get_person_bboxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes of detected persons.
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detections = self.detector.detect(bgr_image, classes=[self.PERSON_CLASS_ID])
        
        bboxes = []
        h, w = image.shape[:2]
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            bboxes.append((x1, y1, x2, y2))
        
        return bboxes

