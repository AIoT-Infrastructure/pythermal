#!/usr/bin/env python3
"""
Environment Temperature Estimation

Provides functions to estimate ambient/environment temperature from thermal frames.
"""

import numpy as np
from typing import Optional

from ..detections.utils import convert_to_celsius


def estimate_environment_temperature(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float,
    percentile: float = 5.0
) -> Optional[float]:
    """
    Estimate environment temperature from a thermal frame.
    
    Version 1: Uses the 5th percentile of frame temperatures as environment temperature.
    This assumes that the coldest pixels represent the ambient/room temperature,
    as they are least affected by heat sources.
    
    Args:
        temp_array: Temperature array (96x96, uint16 raw values or float32 Celsius)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        percentile: Percentile to use for estimation (default: 5.0 for 5th percentile)
    
    Returns:
        Estimated environment temperature in Celsius, or None if input is invalid
    """
    if temp_array is None or temp_array.size == 0:
        return None
    
    # Convert to Celsius if needed
    temp_celsius = convert_to_celsius(temp_array, min_temp, max_temp)
    
    # Flatten array to get all temperature values
    flat_temps = temp_celsius.flatten()
    
    # Calculate the specified percentile (default: 5th percentile)
    # This gives us the temperature below which 5% of pixels fall
    env_temp = np.percentile(flat_temps, percentile)
    
    return float(env_temp)


def estimate_environment_temperature_v1(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float
) -> Optional[float]:
    """
    Estimate environment temperature using version 1 method.
    
    Alias for estimate_environment_temperature with percentile=5.0
    
    Args:
        temp_array: Temperature array (96x96, uint16 raw values or float32 Celsius)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
    
    Returns:
        Estimated environment temperature in Celsius (5th percentile), or None if input is invalid
    """
    return estimate_environment_temperature(temp_array, min_temp, max_temp, percentile=5.0)

