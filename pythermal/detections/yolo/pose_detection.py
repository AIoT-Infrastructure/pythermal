#!/usr/bin/env python3
"""
YOLO v11 Pose Detection for Thermal Images

Provides pose/keypoint detection using YOLO v11 pose models.
Supports both default official models and custom thermal-specific models.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

# Lazy import: ultralytics is only imported when YOLOPoseDetector is instantiated
# This prevents network checks during module import
ULTRALYTICS_AVAILABLE = None
YOLO = None


class YOLOPoseDetector:
    """
    YOLO v11 Pose Detector for thermal images.
    
    Supports both default official YOLO v11 pose models and custom thermal-specific models.
    Models are automatically downloaded on first use if not present.
    """
    
    # Default official YOLO v11 pose model
    DEFAULT_MODEL = "yolo11n-pose.pt"  # nano version for edge devices
    
    # Model size options (nano, small, medium, large, extra-large)
    MODEL_OPTIONS = {
        "nano": "yolo11n-pose.pt",
        "small": "yolo11s-pose.pt",
        "medium": "yolo11m-pose.pt",
        "large": "yolo11l-pose.pt",
        "xlarge": "yolo11x-pose.pt",
    }
    
    # COCO pose keypoint names (17 keypoints)
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    
    # Keypoint connections for skeleton drawing
    SKELETON_CONNECTIONS = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
        # Torso
        (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders-hips
        # Left arm
        (5, 7), (7, 9),  # left shoulder-elbow-wrist
        # Right arm
        (6, 8), (8, 10),  # right shoulder-elbow-wrist
        # Left leg
        (11, 13), (13, 15),  # left hip-knee-ankle
        # Right leg
        (12, 14), (14, 16),  # right hip-knee-ankle
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "nano",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize YOLO pose detector.
        
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
        # This prevents network checks during module import
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
                "ultralytics package is required for YOLO pose detection. "
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
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detect poses/keypoints in thermal image.
        
        Args:
            image: Input image (BGR or RGB, numpy array)
            verbose: Whether to print detection details
        
        Returns:
            List of pose detection dictionaries, each containing:
            - "bbox": [x1, y1, x2, y2] bounding box coordinates
            - "confidence": Detection confidence (0.0-1.0)
            - "keypoints": List of (x, y, confidence) tuples for each keypoint (17 keypoints)
            - "keypoints_dict": Dictionary mapping keypoint names to (x, y, confidence)
            - "center": (x, y) center coordinates of person
            - "width": Bounding box width
            - "height": Bounding box height
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose,
        )
        
        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get keypoints
                person_keypoints = keypoints.xy[i].cpu().numpy()  # [17, 2] (x, y)
                person_keypoint_conf = keypoints.conf[i].cpu().numpy()  # [17] confidence
                
                # Format keypoints as list of (x, y, confidence)
                keypoint_list = []
                keypoint_dict = {}
                for j, (kp_name, (x, y), conf) in enumerate(
                    zip(self.KEYPOINT_NAMES, person_keypoints, person_keypoint_conf)
                ):
                    keypoint_list.append((float(x), float(y), float(conf)))
                    keypoint_dict[kp_name] = (float(x), float(y), float(conf))
                
                # Calculate center and dimensions
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": confidence,
                    "keypoints": keypoint_list,
                    "keypoints_dict": keypoint_dict,
                    "center": (center_x, center_y),
                    "width": width,
                    "height": height,
                })
        
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect poses in multiple images (batch processing).
        
        Args:
            images: List of input images
            verbose: Whether to print detection details
        
        Returns:
            List of pose detection lists (one per image)
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose,
        )
        
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    person_keypoints = keypoints.xy[i].cpu().numpy()
                    person_keypoint_conf = keypoints.conf[i].cpu().numpy()
                    
                    keypoint_list = []
                    keypoint_dict = {}
                    for j, (kp_name, (x, y), conf) in enumerate(
                        zip(self.KEYPOINT_NAMES, person_keypoints, person_keypoint_conf)
                    ):
                        keypoint_list.append((float(x), float(y), float(conf)))
                        keypoint_dict[kp_name] = (float(x), float(y), float(conf))
                    
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        "bbox": box.tolist(),
                        "confidence": confidence,
                        "keypoints": keypoint_list,
                        "keypoints_dict": keypoint_dict,
                        "center": (center_x, center_y),
                        "width": width,
                        "height": height,
                    })
            
            all_detections.append(detections)
        
        return all_detections
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_bbox: bool = True,
        show_keypoints: bool = True,
        show_skeleton: bool = True,
        show_labels: bool = False,
        keypoint_radius: int = 3,
        skeleton_thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize pose detections on image.
        
        Args:
            image: Input image (BGR format)
            detections: List of pose detection dictionaries from detect()
            show_bbox: Whether to show bounding boxes
            show_keypoints: Whether to show keypoints
            show_skeleton: Whether to show skeleton connections
            show_labels: Whether to show keypoint labels
            keypoint_radius: Radius of keypoint circles
            skeleton_thickness: Thickness of skeleton lines
        
        Returns:
            Image with pose detections drawn (BGR format)
        """
        vis_image = image.copy()
        
        # Color palette for keypoints (BGR format)
        keypoint_colors = [
            (255, 0, 0),    # nose - red
            (255, 85, 0),   # left_eye - orange
            (255, 170, 0),  # right_eye - yellow-orange
            (255, 255, 0),  # left_ear - yellow
            (170, 255, 0),  # right_ear - yellow-green
            (85, 255, 0),   # left_shoulder - green
            (0, 255, 0),    # right_shoulder - bright green
            (0, 255, 85),   # left_elbow - green-cyan
            (0, 255, 170),  # right_elbow - cyan
            (0, 255, 255),  # left_wrist - cyan-blue
            (0, 170, 255), # right_wrist - blue-cyan
            (0, 85, 255),  # left_hip - blue
            (0, 0, 255),    # right_hip - bright blue
            (85, 0, 255),   # left_knee - blue-purple
            (170, 0, 255),  # right_knee - purple
            (255, 0, 255),  # left_ankle - magenta
            (255, 0, 170),  # right_ankle - pink
        ]
        
        # Skeleton connection colors (use average of connected keypoint colors)
        skeleton_colors = [
            (255, 42, 0),   # head connections
            (255, 127, 0),  # head connections
            (255, 212, 0),  # head connections
            (170, 255, 0),  # head connections
            (42, 255, 0),   # torso
            (42, 255, 85),  # torso
            (0, 127, 255),  # torso
            (42, 127, 255), # torso
            (42, 255, 42),  # left arm
            (0, 212, 255),  # left arm
            (0, 255, 42),   # right arm
            (0, 212, 255),  # right arm
            (42, 127, 255), # left leg
            (170, 0, 255),  # left leg
            (85, 0, 255),   # right leg
            (170, 0, 255),  # right leg
        ]
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            confidence = det["confidence"]
            keypoints = det["keypoints"]
            
            # Draw bounding box
            if show_bbox:
                bbox_color = (0, 255, 0)  # Green
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)
                
                # Draw confidence score
                if show_labels:
                    conf_text = f"{confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        conf_text, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        vis_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        bbox_color,
                        -1,
                    )
                    cv2.putText(
                        vis_image,
                        conf_text,
                        (x1, y1 - baseline - 2),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                    )
            
            # Draw skeleton connections
            if show_skeleton:
                for idx, (kp1_idx, kp2_idx) in enumerate(self.SKELETON_CONNECTIONS):
                    kp1 = keypoints[kp1_idx]
                    kp2 = keypoints[kp2_idx]
                    
                    # Only draw if both keypoints are visible (confidence > 0)
                    if kp1[2] > 0 and kp2[2] > 0:
                        pt1 = (int(kp1[0]), int(kp1[1]))
                        pt2 = (int(kp2[0]), int(kp2[1]))
                        color = skeleton_colors[idx % len(skeleton_colors)]
                        cv2.line(vis_image, pt1, pt2, color, skeleton_thickness)
            
            # Draw keypoints
            if show_keypoints:
                for idx, (x, y, conf) in enumerate(keypoints):
                    if conf > 0:  # Only draw visible keypoints
                        pt = (int(x), int(y))
                        color = keypoint_colors[idx % len(keypoint_colors)]
                        cv2.circle(vis_image, pt, keypoint_radius, color, -1)
                        
                        # Draw keypoint label
                        if show_labels:
                            kp_name = self.KEYPOINT_NAMES[idx]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.3
                            thickness = 1
                            cv2.putText(
                                vis_image,
                                kp_name,
                                (pt[0] + 5, pt[1] - 5),
                                font,
                                font_scale,
                                color,
                                thickness,
                            )
        
        return vis_image

