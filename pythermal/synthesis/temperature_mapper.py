#!/usr/bin/env python3
"""
Temperature Mapping for Synthetic Thermal Generation

Assigns temperature values to different body parts based on YOLO pose estimation.
Different temperatures are assigned to body parts vs. clothing.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from ..detections.yolo.pose_detection import YOLOPoseDetector
from ..detections.yolo.segmentation import YOLOSegmentationDetector
from ..utils.environment import estimate_body_temperature


class TemperatureMapper:
    """
    Maps temperature values to image regions based on pose keypoints.
    
    Assigns different temperatures to:
    - Body parts (head, torso, limbs) - warmer
    - Clothing - cooler
    - Background - ambient temperature
    """
    
    # Temperature ranges (Celsius)
    DEFAULT_CORE_TEMP = 37.0  # Core body temperature
    DEFAULT_CLOTHING_TEMP = 28.0  # Clothing temperature
    DEFAULT_AMBIENT_TEMP = 22.0  # Room temperature
    
    # Alpha values for different body parts (blood flow regulation coefficient)
    # Used with estimate_body_temperature function
    ALPHA_VALUES = {
        "head": 0.65,  # Face/head: α ≈ 0.5–0.7 (use 0.65 for slightly warmer)
        "torso": 0.6,  # Torso: α ≈ 0.5–0.7 (default: 0.6)
        "limbs": 0.4,  # Limbs: α ≈ 0.3–0.5 (use 0.4)
        "hands_feet": 0.25,  # Extremities: α ≈ 0.2–0.4 (use 0.25)
    }
    
    def __init__(
        self,
        core_temp: float = DEFAULT_CORE_TEMP,
        clothing_temp: float = DEFAULT_CLOTHING_TEMP,
        ambient_temp: float = DEFAULT_AMBIENT_TEMP,
        seg_model_size: str = "nano",
        pose_model_size: str = "nano",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Initialize temperature mapper.
        
        Args:
            core_temp: Core body temperature in Celsius (default: 37.0)
            clothing_temp: Clothing temperature in Celsius (default: 28.0)
            ambient_temp: Ambient/background temperature in Celsius (default: 22.0)
                        Body temperatures will be estimated from this using estimate_body_temperature()
            seg_model_size: YOLO segmentation model size (default: "nano")
            pose_model_size: YOLO pose model size (for body part identification)
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on
        """
        self.core_temp = core_temp
        self.clothing_temp = clothing_temp
        self.ambient_temp = ambient_temp
        
        # Estimate body temperatures from ambient temperature using physiological model
        self.body_temp_head = estimate_body_temperature(
            ambient_temp, alpha=self.ALPHA_VALUES["head"], core_temp=core_temp
        )
        self.body_temp_torso = estimate_body_temperature(
            ambient_temp, alpha=self.ALPHA_VALUES["torso"], core_temp=core_temp
        )
        self.body_temp_limbs = estimate_body_temperature(
            ambient_temp, alpha=self.ALPHA_VALUES["limbs"], core_temp=core_temp
        )
        self.body_temp_extremities = estimate_body_temperature(
            ambient_temp, alpha=self.ALPHA_VALUES["hands_feet"], core_temp=core_temp
        )
        # Skin temperature (exposed skin, typically face/hands) uses head alpha
        self.skin_temp = self.body_temp_head
        
        # Use segmentation model for precise masks
        self.seg_detector = YOLOSegmentationDetector(
            model_size=seg_model_size,
            conf_threshold=conf_threshold,
            device=device,
        )
        
        # Use pose detector for body part identification
        self.pose_detector = YOLOPoseDetector(
            model_size=pose_model_size,
            conf_threshold=conf_threshold,
            device=device,
        )
    
    def create_temperature_map(
        self,
        image: np.ndarray,
        human_mask: np.ndarray,
        pose_detections: Optional[List[Dict[str, Any]]] = None,
        seg_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temperature map from RGB image with human segmentation.
        Uses YOLO segmentation masks for precise temperature assignment.
        
        Args:
            image: RGB image (240, 240, 3)
            human_mask: Binary mask (240, 240) where 255 indicates human pixels
            pose_detections: Optional pre-computed pose detections
            seg_detections: Optional pre-computed segmentation detections
        
        Returns:
            Tuple of:
            - temp_map_240: Temperature map at 240x240 resolution (float32, Celsius)
            - temp_map_96: Temperature map at 96x96 resolution (float32, Celsius)
        """
        h, w = image.shape[:2]
        
        # Initialize temperature map with ambient temperature
        temp_map_240 = np.full((h, w), self.ambient_temp, dtype=np.float32)
        
        # If no human mask, return ambient temperature
        if human_mask.sum() == 0:
            temp_map_96 = cv2.resize(temp_map_240, (96, 96), interpolation=cv2.INTER_LINEAR)
            return temp_map_240, temp_map_96
        
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get segmentation detections if not provided
        if seg_detections is None:
            seg_detections = self.seg_detector.detect(bgr_image, classes=[0])  # Person class ID = 0
        
        # Get pose detections if not provided
        if pose_detections is None:
            pose_detections = self.pose_detector.detect(bgr_image)
        
        # Process each detected person using segmentation masks
        combined_temp_map = np.full((h, w), self.clothing_temp, dtype=np.float32)
        
        # Match segmentation detections with pose detections based on mask overlap
        for seg_det in seg_detections:
            seg_mask = seg_det.get("mask")
            if seg_mask is None:
                continue
            
            # Find matching pose detection (by keypoints within segmentation mask)
            matching_pose = None
            seg_mask_bool = (seg_mask > 0)
            
            for pose_det in pose_detections:
                keypoints_dict = pose_det.get("keypoints_dict", {})
                if not keypoints_dict:
                    continue
                
                # Check if keypoints fall within this segmentation mask
                keypoints_in_mask = 0
                total_keypoints = 0
                for kp_name, kp_data in keypoints_dict.items():
                    if kp_data and kp_data[2] > 0.3:  # Confidence threshold
                        total_keypoints += 1
                        kp_x, kp_y = int(kp_data[0]), int(kp_data[1])
                        if 0 <= kp_y < h and 0 <= kp_x < w:
                            if seg_mask_bool[kp_y, kp_x]:
                                keypoints_in_mask += 1
                
                # If majority of keypoints are in this mask, it's a match
                if total_keypoints > 0 and keypoints_in_mask / total_keypoints > 0.3:
                    matching_pose = pose_det
                    break
            
            # Create person-specific temperature map using segmentation mask
            keypoints_dict = matching_pose.get("keypoints_dict", {}) if matching_pose else {}
            person_temp_map = self._assign_temperatures_to_person_seg(
                image, seg_mask, keypoints_dict
            )
            
            # Update combined map (take maximum temperature where multiple people overlap)
            combined_temp_map = np.maximum(combined_temp_map, person_temp_map)
        
        # Update temperature map where humans exist
        person_mask = (human_mask > 0).astype(np.float32)
        temp_map_240 = temp_map_240 * (1 - person_mask) + combined_temp_map * person_mask
        
        # Downsample to 96x96 for temperature array
        temp_map_96 = cv2.resize(temp_map_240, (96, 96), interpolation=cv2.INTER_LINEAR)
        
        return temp_map_240, temp_map_96
    
    def _assign_temperatures_to_person_seg(
        self,
        image: np.ndarray,
        seg_mask: np.ndarray,
        keypoints_dict: Dict[str, Tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Assign temperatures to a single person using segmentation mask.
        Uses precise segmentation mask instead of keypoint-based shapes.
        
        Args:
            image: RGB image
            seg_mask: Binary segmentation mask (H, W) where 255 indicates person pixels
            keypoints_dict: Dictionary mapping keypoint names to (x, y, confidence)
        
        Returns:
            Temperature map for this person (only non-zero within mask)
        """
        h, w = image.shape[:2]
        # Initialize with ambient temperature
        temp_map = np.full((h, w), self.ambient_temp, dtype=np.float32)
        
        # Get mask boolean
        mask_bool = (seg_mask > 0)
        
        if not mask_bool.any():
            return temp_map
        
        # Start with clothing temperature for entire mask
        temp_map[mask_bool] = self.clothing_temp
        
        # Use keypoints to identify body parts and assign higher temperatures
        if keypoints_dict:
            # Get keypoint coordinates
            def get_kp(name: str) -> Optional[Tuple[float, float]]:
                kp = keypoints_dict.get(name)
                if kp and kp[2] > 0.3:  # Confidence threshold
                    return (kp[0], kp[1])
                return None
            
            # Head region - use adaptive radius based on segmentation mask
            head_kps = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
            head_points = [get_kp(kp) for kp in head_kps if get_kp(kp) is not None]
            
            if head_points:
                head_center = np.mean([p for p in head_points if p is not None], axis=0)
                # Estimate head region from segmentation mask in upper portion
                head_radius = self._estimate_head_radius_from_mask(seg_mask, head_center, h)
                head_mask = self._create_circle_mask(h, w, head_center, head_radius) & mask_bool
                # Use estimated head temperature (warmer, alpha=0.65)
                temp_map[head_mask] = self.body_temp_head
            
            # Torso region - connect shoulders to hips and fill within mask
            shoulder_kps = ["left_shoulder", "right_shoulder"]
            hip_kps = ["left_hip", "right_hip"]
            shoulders = [get_kp(kp) for kp in shoulder_kps if get_kp(kp) is not None]
            hips = [get_kp(kp) for kp in hip_kps if get_kp(kp) is not None]
            
            if shoulders and hips:
                torso_mask = self._create_torso_mask_from_keypoints(
                    h, w, shoulders, hips, mask_bool
                )
                # Use estimated torso temperature (alpha=0.6)
                temp_map[torso_mask] = self.body_temp_torso
            
            # Limbs - create connected regions between joints
            # Left arm: shoulder -> elbow -> wrist
            left_shoulder = get_kp("left_shoulder")
            left_elbow = get_kp("left_elbow")
            left_wrist = get_kp("left_wrist")
            if left_shoulder and left_elbow and left_wrist:
                left_arm_mask = self._create_limb_mask(
                    h, w, [left_shoulder, left_elbow, left_wrist], mask_bool, radius=12
                )
                # Use estimated limb temperature (alpha=0.4)
                temp_map[left_arm_mask] = self.body_temp_limbs
            
            # Right arm: shoulder -> elbow -> wrist
            right_shoulder = get_kp("right_shoulder")
            right_elbow = get_kp("right_elbow")
            right_wrist = get_kp("right_wrist")
            if right_shoulder and right_elbow and right_wrist:
                right_arm_mask = self._create_limb_mask(
                    h, w, [right_shoulder, right_elbow, right_wrist], mask_bool, radius=12
                )
                # Use estimated limb temperature (alpha=0.4)
                temp_map[right_arm_mask] = self.body_temp_limbs
            
            # Left leg: hip -> knee -> ankle
            left_hip = get_kp("left_hip")
            left_knee = get_kp("left_knee")
            left_ankle = get_kp("left_ankle")
            if left_hip and left_knee and left_ankle:
                left_leg_mask = self._create_limb_mask(
                    h, w, [left_hip, left_knee, left_ankle], mask_bool, radius=12
                )
                # Use estimated limb temperature (alpha=0.4)
                temp_map[left_leg_mask] = self.body_temp_limbs
            
            # Right leg: hip -> knee -> ankle
            right_hip = get_kp("right_hip")
            right_knee = get_kp("right_knee")
            right_ankle = get_kp("right_ankle")
            if right_hip and right_knee and right_ankle:
                right_leg_mask = self._create_limb_mask(
                    h, w, [right_hip, right_knee, right_ankle], mask_bool, radius=12
                )
                # Use estimated limb temperature (alpha=0.4)
                temp_map[right_leg_mask] = self.body_temp_limbs
            
            # Extremities - small circles for hands and feet
            extremity_kps = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
            extremity_points = [get_kp(kp) for kp in extremity_kps if get_kp(kp) is not None]
            
            for ext_point in extremity_points:
                if ext_point:
                    # Small radius for extremities
                    ext_mask = self._create_circle_mask(h, w, ext_point, 8) & mask_bool
                    # Use estimated extremity temperature (alpha=0.25, cooler)
                    temp_map[ext_mask] = self.body_temp_extremities
        
        # Differentiate skin from clothing using RGB color analysis
        skin_mask = self._detect_skin_regions(image, mask_bool)
        
        # Assign skin temperature to detected skin regions
        # Override body part temperatures with skin_temp where skin is detected
        # This ensures exposed skin (face, hands, arms) gets proper skin temperature
        skin_regions = skin_mask & mask_bool
        temp_map[skin_regions] = self.skin_temp
        
        # For body parts that are not detected as skin, they're likely covered by clothing
        # Keep their assigned temperatures (estimated from ambient_temp) but ensure they're warmer than pure clothing
        
        # Fill any remaining mask regions that don't have body part temperatures
        # Use distance-based assignment: closer to body parts = warmer
        unassigned_mask = mask_bool & (temp_map == self.clothing_temp)
        if unassigned_mask.any():
            # Assign intermediate temperature based on proximity to body parts
            temp_map = self._fill_gaps_with_proximity(temp_map, mask_bool, unassigned_mask)
        
        # Minimal blurring for high-quality thermal camera (very light smoothing only)
        # Use smaller kernel and lower sigma to preserve texture
        temp_map = cv2.GaussianBlur(temp_map, (3, 3), 0.3)
        
        # Ensure non-mask regions are ambient
        temp_map[~mask_bool] = self.ambient_temp
        
        return temp_map
    
    def _create_circle_mask(self, h: int, w: int, center: Tuple[float, float], radius: float) -> np.ndarray:
        """Create a binary circle mask."""
        y, x = np.ogrid[:h, :w]
        cy, cx = int(center[1]), int(center[0])
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        return mask
    
    def _create_ellipse_mask(self, h: int, w: int, center: Tuple[float, float], size: Tuple[float, float]) -> np.ndarray:
        """Create a binary ellipse mask."""
        mask = np.zeros((h, w), dtype=bool)
        cy, cx = int(center[1]), int(center[0])
        axes = (int(size[0]), int(size[1]))
        cv2.ellipse(mask.astype(np.uint8), (cx, cy), axes, 0, 0, 360, 255, -1)
        return mask.astype(bool)
    
    def _estimate_head_radius(self, head_points: List[Tuple[float, float]], center: np.ndarray) -> float:
        """Estimate head radius from keypoints."""
        if not head_points:
            return 30.0  # Default radius
        
        distances = [np.linalg.norm(np.array(p) - center) for p in head_points]
        return max(25.0, np.mean(distances) * 1.5) if distances else 30.0
    
    def _estimate_torso_size(self, shoulders: List[Tuple[float, float]], hips: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Estimate torso ellipse size."""
        if not shoulders or not hips:
            return (40.0, 60.0)  # Default size
        
        shoulder_center = np.mean([p for p in shoulders if p is not None], axis=0)
        hip_center = np.mean([p for p in hips if p is not None], axis=0)
        
        width = np.linalg.norm(np.array(shoulders[0]) - np.array(shoulders[1])) if len(shoulders) >= 2 else 40.0
        height = np.linalg.norm(shoulder_center - hip_center) if len(hips) > 0 else 60.0
        
        return (max(30.0, width * 0.8), max(40.0, height * 0.9))
    
    def _estimate_head_radius_from_mask(self, seg_mask: np.ndarray, head_center: Tuple[float, float], image_height: int) -> float:
        """Estimate head radius from segmentation mask in upper portion of image."""
        h, w = seg_mask.shape
        cy, cx = int(head_center[1]), int(head_center[0])
        
        # Look at upper 40% of image for head region
        upper_bound = int(h * 0.4)
        head_region = seg_mask[:upper_bound, :]
        
        # Find connected components in head region
        if head_region.sum() > 0:
            # Estimate radius from mask density around head center
            y_range = max(0, cy - 30), min(upper_bound, cy + 30)
            x_range = max(0, cx - 30), min(w, cx + 30)
            local_mask = seg_mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            
            if local_mask.sum() > 0:
                # Find average distance from center
                y_coords, x_coords = np.where(local_mask > 0)
                if len(y_coords) > 0:
                    local_cy, local_cx = cy - y_range[0], cx - x_range[0]
                    distances = np.sqrt((x_coords - local_cx)**2 + (y_coords - local_cy)**2)
                    radius = np.percentile(distances, 75)  # Use 75th percentile
                    return max(20.0, min(radius * 1.2, 40.0))
        
        return 25.0  # Default fallback
    
    def _create_torso_mask_from_keypoints(
        self,
        h: int,
        w: int,
        shoulders: List[Tuple[float, float]],
        hips: List[Tuple[float, float]],
        body_mask: np.ndarray,
    ) -> np.ndarray:
        """Create torso mask connecting shoulders to hips, constrained by segmentation mask."""
        torso_mask = np.zeros((h, w), dtype=bool)
        
        if len(shoulders) >= 2 and len(hips) >= 2:
            # Create convex hull of torso keypoints
            torso_points = np.array(shoulders + hips, dtype=np.int32)
            
            # Create polygon mask
            cv2.fillPoly(torso_mask.astype(np.uint8), [torso_points], 255)
            torso_mask = torso_mask.astype(bool)
            
            # Expand slightly to cover more area
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            torso_mask = cv2.dilate(torso_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            # Constrain to body segmentation mask
            torso_mask = torso_mask & body_mask
        
        return torso_mask
    
    def _create_limb_mask(
        self,
        h: int,
        w: int,
        joint_points: List[Tuple[float, float]],
        body_mask: np.ndarray,
        radius: float = 12.0,
    ) -> np.ndarray:
        """Create mask for a limb connecting multiple joints."""
        limb_mask = np.zeros((h, w), dtype=bool)
        
        if len(joint_points) < 2:
            return limb_mask
        
        # Draw lines between joints
        for i in range(len(joint_points) - 1):
            pt1 = (int(joint_points[i][0]), int(joint_points[i][1]))
            pt2 = (int(joint_points[i+1][0]), int(joint_points[i+1][1]))
            cv2.line(limb_mask.astype(np.uint8), pt1, pt2, 255, int(radius * 2))
        
        # Add circles at joints
        for joint in joint_points:
            joint_mask = self._create_circle_mask(h, w, joint, radius)
            limb_mask = limb_mask | joint_mask
        
        # Constrain to body segmentation mask
        limb_mask = limb_mask & body_mask
        
        return limb_mask
    
    def _fill_gaps_with_proximity(
        self,
        temp_map: np.ndarray,
        body_mask: np.ndarray,
        unassigned_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill gaps in temperature map using proximity to assigned regions."""
        if not unassigned_mask.any():
            return temp_map
        
        # Find regions with body part temperatures (warmer than clothing)
        body_part_mask = (temp_map > self.clothing_temp + 0.5) & body_mask
        
        if not body_part_mask.any():
            return temp_map
        
        # Use distance transform to assign temperatures based on proximity
        # Closer to body parts = warmer
        dist_transform = cv2.distanceTransform(
            (~body_part_mask).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Normalize distance and assign intermediate temperatures
        max_dist = dist_transform.max()
        if max_dist > 0:
            # Invert: closer = higher value
            proximity = 1.0 - (dist_transform / max_dist)
            proximity = np.clip(proximity, 0, 1)
            
            # Assign temperatures based on proximity (between clothing and torso temp)
            temp_range = self.body_temp_torso - self.clothing_temp
            assigned_temp = self.clothing_temp + proximity * temp_range * 0.5  # Max 50% of range
            
            # Only update unassigned regions
            temp_map[unassigned_mask] = assigned_temp[unassigned_mask]
        
        return temp_map
    
    def _detect_skin_regions(self, rgb_image: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        """
        Detect skin regions in RGB image using color analysis.
        
        Uses HSV color space to identify skin-tone colors (flesh tones).
        
        Args:
            rgb_image: RGB image (H, W, 3)
            body_mask: Binary mask indicating person regions
        
        Returns:
            Binary mask (H, W) where True indicates skin pixels
        """
        h, w = rgb_image.shape[:2]
        skin_mask = np.zeros((h, w), dtype=bool)
        
        if not body_mask.any():
            return skin_mask
        
        # Convert RGB to HSV for better skin color detection
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Define skin color ranges in HSV
        # These ranges cover various skin tones (light to dark)
        # Hue: 0-20 (red/orange tones) and 160-180 (pink tones)
        # Saturation: 20-255 (some color, not grayscale)
        # Value: 50-255 (not too dark)
        
        # Lower bound for skin (reddish tones)
        lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_image, lower_skin1, upper_skin1)
        
        # Upper bound for skin (pinkish tones)
        lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_image, lower_skin2, upper_skin2)
        
        # Combine masks
        skin_color_mask = mask1 | mask2
        
        # Additional check: skin typically has higher red component
        # and moderate green/blue components
        r_channel = rgb_image[:, :, 0].astype(np.float32)
        g_channel = rgb_image[:, :, 1].astype(np.float32)
        b_channel = rgb_image[:, :, 2].astype(np.float32)
        
        # Skin has: R > G > B (generally) and R > 95
        red_dominant = (r_channel > g_channel) & (g_channel > b_channel)
        red_sufficient = r_channel > 95
        
        # Combine color-based detection with RGB analysis
        skin_mask = (skin_color_mask > 0) & red_dominant & red_sufficient
        
        # Constrain to body mask
        skin_mask = skin_mask & body_mask
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = skin_mask.astype(bool)
        
        return skin_mask
    
    def _fill_circle(self, arr: np.ndarray, center: Tuple[float, float], radius: float, value: float):
        """Fill circle region in array."""
        h, w = arr.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = int(center[1]), int(center[0])
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        arr[mask] = value
    
    def _fill_ellipse(self, arr: np.ndarray, center: Tuple[float, float], size: Tuple[float, float], value: float):
        """Fill ellipse region in array."""
        h, w = arr.shape
        cy, cx = int(center[1]), int(center[0])
        axes = (int(size[0]), int(size[1]))
        
        # Create ellipse mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        arr[mask > 0] = value

