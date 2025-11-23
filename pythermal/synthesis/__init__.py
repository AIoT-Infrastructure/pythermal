#!/usr/bin/env python3
"""
Synthetic Thermal Data Generation Module

This module provides functionality to generate synthetic thermal frames (.tframe format)
from RGB images. It includes:
- Image preprocessing (resize/crop to 240x240)
- Human segmentation using YOLO
- Pose estimation using YOLO pose detection
- Temperature assignment based on body parts
- Frame generation and export to .tframe format

Designed to be extendable for video synthesis in the future.
"""

from .synthetic_generator import SyntheticThermalGenerator
from .image_processor import ImageProcessor
from .human_segmentation import HumanSegmenter
from .temperature_mapper import TemperatureMapper
from .frame_generator import ThermalFrameGenerator

__all__ = [
    "SyntheticThermalGenerator",
    "ImageProcessor",
    "HumanSegmenter",
    "TemperatureMapper",
    "ThermalFrameGenerator",
]

