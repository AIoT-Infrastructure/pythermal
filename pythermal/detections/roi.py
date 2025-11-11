#!/usr/bin/env python3
"""
Region of Interest (ROI) support for thermal detection.

Provides ROI management and zone monitoring capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .utils import DetectedObject


@dataclass
class ROI:
    """
    Region of Interest definition.
    
    Represents a rectangular region with optional temperature thresholds.
    """
    x: int
    y: int
    width: int
    height: int
    name: str = "ROI"
    temp_min: Optional[float] = None
    temp_max: Optional[float] = None
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) is within this ROI."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)
    
    def get_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary mask for this ROI.
        
        Args:
            image_shape: (height, width) of the image
        
        Returns:
            Binary mask (uint8) where ROI region is 255, rest is 0
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        mask[self.y:self.y+self.height, self.x:self.x+self.width] = 255
        return mask


class ROIManager:
    """
    Manages multiple ROIs for zone monitoring.
    
    Supports multiple ROIs with different temperature thresholds.
    """
    
    def __init__(self, image_width: int = 96, image_height: int = 96):
        """
        Initialize ROI manager.
        
        Args:
            image_width: Width of the thermal image (default: 96)
            image_height: Height of the thermal image (default: 96)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.rois: List[ROI] = []
    
    def add_roi(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        name: str = "ROI",
        temp_min: Optional[float] = None,
        temp_max: Optional[float] = None
    ) -> ROI:
        """
        Add a new ROI.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of ROI
            height: Height of ROI
            name: Name/label for this ROI
            temp_min: Optional minimum temperature threshold for this ROI
            temp_max: Optional maximum temperature threshold for this ROI
        
        Returns:
            The created ROI instance
        """
        roi = ROI(x=x, y=y, width=width, height=height, name=name,
                  temp_min=temp_min, temp_max=temp_max)
        self.rois.append(roi)
        return roi
    
    def add_center_roi(
        self,
        size: int = 30,
        name: str = "Center",
        temp_min: Optional[float] = None,
        temp_max: Optional[float] = None
    ) -> ROI:
        """
        Add a centered ROI.
        
        Args:
            size: Size of the square ROI (default: 30)
            name: Name/label for this ROI
            temp_min: Optional minimum temperature threshold for this ROI
            temp_max: Optional maximum temperature threshold for this ROI
        
        Returns:
            The created ROI instance
        """
        x = (self.image_width - size) // 2
        y = (self.image_height - size) // 2
        return self.add_roi(x, y, size, size, name, temp_min, temp_max)
    
    def remove_roi(self, name: str) -> bool:
        """
        Remove ROI by name.
        
        Args:
            name: Name of ROI to remove
        
        Returns:
            True if ROI was found and removed, False otherwise
        """
        for i, roi in enumerate(self.rois):
            if roi.name == name:
                self.rois.pop(i)
                return True
        return False
    
    def clear(self):
        """Remove all ROIs."""
        self.rois.clear()
    
    def get_combined_mask(self) -> np.ndarray:
        """
        Get combined mask for all ROIs.
        
        Returns:
            Binary mask (uint8) where any ROI region is 255, rest is 0
        """
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for roi in self.rois:
            roi_mask = roi.get_mask((self.image_height, self.image_width))
            mask = np.maximum(mask, roi_mask)
        return mask
    
    def filter_objects_by_roi(
        self,
        objects: List[DetectedObject],
        roi_name: Optional[str] = None
    ) -> List[DetectedObject]:
        """
        Filter detected objects to only include those within ROI(s).
        
        Args:
            objects: List of detected objects
            roi_name: If specified, only filter by this ROI. Otherwise, filter by all ROIs.
        
        Returns:
            Filtered list of objects within ROI(s)
        """
        if not self.rois:
            return objects
        
        filtered = []
        rois_to_check = [r for r in self.rois if roi_name is None or r.name == roi_name]
        
        if not rois_to_check:
            return []
        
        for obj in objects:
            for roi in rois_to_check:
                if roi.contains(obj.center_x, obj.center_y):
                    filtered.append(obj)
                    break
        
        return filtered
    
    def filter_objects_by_temperature(
        self,
        objects: List[DetectedObject],
        temp_celsius: np.ndarray,
        roi_name: Optional[str] = None
    ) -> List[DetectedObject]:
        """
        Filter detected objects by ROI temperature thresholds.
        
        Args:
            objects: List of detected objects
            temp_celsius: Temperature array in Celsius
            roi_name: If specified, only check this ROI. Otherwise, check all ROIs.
        
        Returns:
            Filtered list of objects that meet ROI temperature criteria
        """
        if not self.rois:
            return objects
        
        filtered = []
        rois_to_check = [r for r in self.rois if roi_name is None or r.name == roi_name]
        
        if not rois_to_check:
            return []
        
        for obj in objects:
            # Find which ROI(s) contain this object
            for roi in rois_to_check:
                if roi.contains(obj.center_x, obj.center_y):
                    # Check temperature thresholds if specified
                    if roi.temp_min is not None and obj.avg_temperature < roi.temp_min:
                        continue
                    if roi.temp_max is not None and obj.avg_temperature > roi.temp_max:
                        continue
                    filtered.append(obj)
                    break
        
        return filtered
    
    def get_roi_statistics(
        self,
        temp_celsius: np.ndarray,
        roi_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get temperature statistics for each ROI.
        
        Args:
            temp_celsius: Temperature array in Celsius
            roi_name: If specified, only get stats for this ROI. Otherwise, get stats for all ROIs.
        
        Returns:
            Dictionary mapping ROI names to their statistics (min, max, avg)
        """
        stats = {}
        rois_to_check = [r for r in self.rois if roi_name is None or r.name == roi_name]
        
        for roi in rois_to_check:
            roi_mask = roi.get_mask((self.image_height, self.image_width))
            roi_temps = temp_celsius[roi_mask > 0]
            
            if len(roi_temps) > 0:
                stats[roi.name] = {
                    'min': float(np.min(roi_temps)),
                    'max': float(np.max(roi_temps)),
                    'avg': float(np.mean(roi_temps)),
                    'count': len(roi_temps)
                }
            else:
                stats[roi.name] = {
                    'min': 0.0,
                    'max': 0.0,
                    'avg': 0.0,
                    'count': 0
                }
        
        return stats

