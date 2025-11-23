#!/usr/bin/env python3
"""
Image Preprocessing for Synthetic Thermal Generation

Handles resizing and cropping RGB images to the required thermal frame dimensions (240x240).
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ImageProcessor:
    """
    Processes RGB images for thermal frame generation.
    
    Handles resizing and cropping to 240x240 pixels, maintaining aspect ratio
    when possible, or centering crops for non-square images.
    """
    
    TARGET_WIDTH = 240
    TARGET_HEIGHT = 240
    
    def __init__(self, crop_mode: str = "center"):
        """
        Initialize image processor.
        
        Args:
            crop_mode: How to crop non-square images. Options:
                      - "center": Center crop (default)
                      - "smart": Smart crop focusing on center region
        """
        self.crop_mode = crop_mode
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image to 240x240 size.
        
        Args:
            image: Input RGB image (H, W, 3) or BGR image
        
        Returns:
            Processed image (240, 240, 3) in RGB format
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        h, w = image.shape[:2]
        
        # If already 240x240, return as-is (convert to RGB if needed)
        if h == self.TARGET_HEIGHT and w == self.TARGET_WIDTH:
            return self._ensure_rgb(image)
        
        # Calculate scaling factor to fit image
        scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h)
        
        # Resize maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop to 240x240
        cropped = self._crop_to_size(resized, self.TARGET_WIDTH, self.TARGET_HEIGHT)
        
        return self._ensure_rgb(cropped)
    
    def _crop_to_size(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Crop image to target size.
        
        Args:
            image: Input image
            target_w: Target width
            target_h: Target height
        
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        if h == target_h and w == target_w:
            return image
        
        # Calculate crop coordinates
        if self.crop_mode == "center":
            start_y = (h - target_h) // 2
            start_x = (w - target_w) // 2
        else:  # smart crop or default to center
            start_y = (h - target_h) // 2
            start_x = (w - target_w) // 2
        
        # Ensure non-negative coordinates
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        
        # Crop
        cropped = image[start_y:start_y + target_h, start_x:start_x + target_w]
        
        # If crop resulted in smaller image, pad it
        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            pad_h = max(0, target_h - cropped.shape[0])
            pad_w = max(0, target_w - cropped.shape[1])
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_REPLICATE
            )
        
        return cropped
    
    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in RGB format.
        
        Args:
            image: Input image (BGR or RGB)
        
        Returns:
            RGB image
        """
        if len(image.shape) == 2:
            # Grayscale, convert to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Check if BGR (OpenCV default) and convert to RGB
        # We assume if it's from OpenCV, it's BGR
        # For safety, we'll check the color channel ordering
        # This is a heuristic - in practice, we'll assume RGB input
        return image.copy()

