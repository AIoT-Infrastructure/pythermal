#!/usr/bin/env python3
"""
YOLO v11 Segmentation Detection for Thermal Images

Provides instance segmentation using YOLO v11 segmentation models.
Supports both default official models and custom thermal-specific models.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

# Lazy import: ultralytics is only imported when YOLOSegmentationDetector is instantiated
ULTRALYTICS_AVAILABLE = None
YOLO = None


class YOLOSegmentationDetector:
    """
    YOLO v11 Segmentation Detector for thermal images.
    
    Supports both default official YOLO v11 segmentation models and custom thermal-specific models.
    Models are automatically downloaded on first use if not present.
    """
    
    # Default official YOLO v11 segmentation model
    DEFAULT_MODEL = "yolo11n-seg.pt"  # nano version for edge devices
    
    # Model size options (nano, small, medium, large, extra-large)
    MODEL_OPTIONS = {
        "nano": "yolo11n-seg.pt",
        "small": "yolo11s-seg.pt",
        "medium": "yolo11m-seg.pt",
        "large": "yolo11l-seg.pt",
        "xlarge": "yolo11x-seg.pt",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "nano",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize YOLO segmentation detector.
        
        Args:
            model_path: Path to custom model file. If None, uses default official model.
                       Can be absolute path or relative to models directory.
            model_size: Model size for default model ("nano", "small", "medium", "large", "xlarge").
                        Only used if model_path is None.
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            device: Device to run inference on ("cpu", "cuda", "mps", etc.). 
                   If None, auto-detects.
        
        Raises:
            ImportError: If ultralytics package is not installed
            FileNotFoundError: If custom model file is not found
        """
        # Lazy import ultralytics only when actually instantiating the detector
        global ULTRALYTICS_AVAILABLE, YOLO
        if ULTRALYTICS_AVAILABLE is None:
            try:
                from ultralytics import YOLO
                ULTRALYTICS_AVAILABLE = True
            except ImportError:
                ULTRALYTICS_AVAILABLE = False
                YOLO = None
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics package is required for YOLO segmentation. "
                "Install it with: pip install ultralytics"
            )
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Determine model path
        if model_path is None:
            # Use default official model
            model_name = self.MODEL_OPTIONS.get(model_size.lower(), self.DEFAULT_MODEL)
            self.model_path = model_name  # ultralytics will download it automatically
            self.is_custom_model = False
        else:
            # Use custom model
            self.model_path = self._resolve_model_path(model_path)
            self.is_custom_model = True
        
        # Load model
        self.model = YOLO(self.model_path)
        
        # Set device if specified
        if self.device is not None:
            self.model.to(self.device)
    
    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve model path - check if it's absolute, relative to models dir, or just filename.
        
        Args:
            model_path: Model path provided by user
        
        Returns:
            Resolved absolute path to model file
        
        Raises:
            FileNotFoundError: If model file is not found
        """
        # If absolute path and exists, use it
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path
        
        # Get models directory
        models_dir = Path(__file__).parent / "models"
        
        # Check if file exists in models directory
        model_file = models_dir / model_path
        if model_file.exists():
            return str(model_file)
        
        # Check if it's just a filename and exists in models dir
        if os.path.exists(model_path):
            return model_path
        
        # Model not found
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Checked:\n"
            f"  - Absolute path: {model_path}\n"
            f"  - Models directory: {model_file}\n"
            f"  - Current directory: {os.path.abspath(model_path)}\n"
            f"Please ensure the model file exists or use a default model."
        )
    
    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detect and segment objects in image.
        
        Args:
            image: Input image (BGR or RGB, numpy array)
            classes: List of class IDs to detect (None = all classes)
            verbose: Whether to print detection details
        
        Returns:
            List of segmentation detection dictionaries, each containing:
            - "bbox": [x1, y1, x2, y2] bounding box coordinates
            - "confidence": Detection confidence (0.0-1.0)
            - "class_id": Class ID
            - "class_name": Class name
            - "mask": Binary segmentation mask (H, W) where 255 indicates object pixels
            - "center": (x, y) center coordinates
            - "width": Bounding box width
            - "height": Bounding box height
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=verbose,
        )
        
        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks
            
            h, w = image.shape[:2]
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Get segmentation mask
                mask = None
                if masks is not None and i < len(masks.data):
                    # Get mask data
                    mask_data = masks.data[i].cpu().numpy()  # Shape: (H_mask, W_mask)
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Convert to binary mask (0 or 255)
                    mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Calculate center and dimensions
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "mask": mask,
                    "center": (center_x, center_y),
                    "width": width,
                    "height": height,
                })
        
        return detections

