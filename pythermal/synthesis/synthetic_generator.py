#!/usr/bin/env python3
"""
Synthetic Thermal Frame Generator

Main API for converting RGB images to synthetic thermal frames (.tframe format).
Coordinates image processing, human segmentation, pose estimation, temperature mapping,
and frame generation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
from .image_processor import ImageProcessor
from .human_segmentation import HumanSegmenter
from .temperature_mapper import TemperatureMapper
from .frame_generator import ThermalFrameGenerator
from ..core.frame_processor import ThermalFrameProcessor, ThermalFrame


class SyntheticThermalGenerator:
    """
    Main class for generating synthetic thermal frames from RGB images.
    
    Pipeline:
    1. Resize/crop image to 240x240
    2. Segment humans from image
    3. Use YOLO pose estimation to identify body parts
    4. Assign different temperatures to body parts and clothing
    5. Generate ThermalFrame and export as .tframe
    """
    
    def __init__(
        self,
        core_temp: float = 37.0,
        clothing_temp: float = 28.0,
        ambient_temp: float = 22.0,
        model_size: str = "nano",
        seg_model_size: str = "nano",
        pose_model_size: str = "nano",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Initialize synthetic thermal generator.
        
        Args:
            core_temp: Core body temperature in Celsius (default: 37.0)
                      Body part temperatures are estimated from ambient_temp using
                      estimate_body_temperature() with different alpha values.
            clothing_temp: Clothing temperature in Celsius (default: 28.0)
            ambient_temp: Ambient/background temperature in Celsius (default: 22.0)
                         Body temperatures are estimated from this using physiological model
            model_size: YOLO object detection model size (default: "nano")
            seg_model_size: YOLO segmentation model size (default: "nano")
            pose_model_size: YOLO pose detection model size (default: "nano")
            conf_threshold: Confidence threshold for detections (default: 0.25)
            device: Device to run inference on ("cpu", "cuda", etc.)
        """
        self.image_processor = ImageProcessor()
        self.human_segmenter = HumanSegmenter(
            model_size=model_size,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.temperature_mapper = TemperatureMapper(
            core_temp=core_temp,
            clothing_temp=clothing_temp,
            ambient_temp=ambient_temp,
            seg_model_size=seg_model_size,
            pose_model_size=pose_model_size,
            conf_threshold=conf_threshold,
            device=device,
        )
        # Estimate max body temp for frame generator (use head temp as max)
        from ..utils.environment import estimate_body_temperature
        max_body_temp = estimate_body_temperature(ambient_temp, alpha=0.65, core_temp=core_temp)
        self.frame_generator = ThermalFrameGenerator(
            temp_min=min(ambient_temp - 5, 15.0),
            temp_max=max(max_body_temp + 5, 45.0),
        )
    
    def generate_from_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        view_mode: str = "temperature",
        sequence: int = 0,
        timestamp: Optional[float] = None,
    ) -> Tuple[ThermalFrame, np.ndarray]:
        """
        Generate synthetic thermal frame from RGB image.
        
        Args:
            image_path: Path to input RGB image
            output_path: Optional path to save .tframe file
            view_mode: View mode for rendered image ("yuyv", "temperature", "temperature_celsius")
            sequence: Frame sequence number (default: 0)
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            Tuple of:
            - ThermalFrame object
            - Rendered image (BGR format) with overlay
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.generate_from_array(
            rgb_image,
            output_path=output_path,
            view_mode=view_mode,
            sequence=sequence,
            timestamp=timestamp,
        )
    
    def generate_from_array(
        self,
        rgb_image: np.ndarray,
        output_path: Optional[str] = None,
        view_mode: str = "temperature",
        sequence: int = 0,
        timestamp: Optional[float] = None,
    ) -> Tuple[ThermalFrame, np.ndarray]:
        """
        Generate synthetic thermal frame from RGB numpy array.
        
        Args:
            rgb_image: RGB image array (any size, will be resized to 240x240)
            output_path: Optional path to save .tframe file
            view_mode: View mode for rendered image
            sequence: Frame sequence number
            timestamp: Optional timestamp
        
        Returns:
            Tuple of:
            - ThermalFrame object
            - Rendered image (BGR format) with overlay
        """
        # Step 1: Process image (resize/crop to 240x240)
        processed_image = self.image_processor.process(rgb_image)
        
        # Step 2: Segment humans (get segmentation masks)
        human_mask, seg_detections = self.human_segmenter.segment(processed_image)
        
        # Step 3: Get pose detections
        bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        pose_detections = self.temperature_mapper.pose_detector.detect(bgr_image)
        
        # Step 4: Create temperature map using segmentation masks
        temp_map_240, temp_map_96 = self.temperature_mapper.create_temperature_map(
            processed_image,
            human_mask,
            pose_detections=pose_detections,
            seg_detections=seg_detections,
        )
        
        # Step 5: Generate ThermalFrame
        thermal_frame = self.frame_generator.generate_frame(
            processed_image,
            temp_map_96,
            timestamp=timestamp,
            sequence=sequence,
        )
        
        # Step 6: Create rendered image
        rendered_image = self._create_rendered_image(
            processed_image,
            temp_map_240,
            thermal_frame,
            view_mode,
            pose_detections,
        )
        
        # Step 7: Save to .tframe if output path provided
        if output_path:
            ThermalFrameProcessor.write_tframe(
                output_path,
                rendered_image,
                thermal_frame,
                view_mode=view_mode,
            )
        
        return thermal_frame, rendered_image
    
    def _create_rendered_image(
        self,
        rgb_image: np.ndarray,
        temp_map_240: np.ndarray,
        thermal_frame: ThermalFrame,
        view_mode: str,
        pose_detections: list,
    ) -> np.ndarray:
        """
        Create rendered image with overlay for .tframe file.
        
        Args:
            rgb_image: RGB image (240, 240, 3)
            temp_map_240: Temperature map (240, 240)
            thermal_frame: ThermalFrame object
            view_mode: View mode string
            pose_detections: List of pose detections
        
        Returns:
            Rendered BGR image with overlay
        """
        # Convert temperature map to visualization
        if view_mode == "temperature" or view_mode == "temperature_celsius":
            # Normalize temperature map to 0-255
            temp_min = thermal_frame.metadata.min_temp
            temp_max = thermal_frame.metadata.max_temp
            temp_range = temp_max - temp_min
            
            if temp_range > 0:
                normalized = ((temp_map_240 - temp_min) / temp_range * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(temp_map_240, dtype=np.uint8)
            
            # Apply colormap
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
            image = colored
        else:
            # YUYV view - convert RGB to grayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Add text overlay with metadata
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        info_text = [
            f"Seq: {thermal_frame.metadata.seq}",
            f"Min: {thermal_frame.metadata.min_temp:.1f}C",
            f"Max: {thermal_frame.metadata.max_temp:.1f}C",
            f"Avg: {thermal_frame.metadata.avg_temp:.1f}C",
        ]
        
        y_offset = 20
        for text in info_text:
            cv2.putText(image, text, (5, y_offset), font, font_scale, color, thickness)
            y_offset += 20
        
        # Note: Skeleton visualization removed from .tframe rendered image
        # to keep the thermal visualization clean
        
        return image

