#!/usr/bin/env python3
"""
Environment Temperature Estimation

Provides functions to estimate ambient/environment temperature from thermal frames.
"""

import numpy as np
from typing import Optional

from ..detections.utils import convert_to_celsius

# Global state for smoothing filter
_env_temp_history = []
_env_temp_smoothed = None
_MAX_HISTORY_SIZE = 10


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
    max_temp: float,
    smoothing: bool = True,
    smoothing_factor: float = 0.3
) -> Optional[float]:
    """
    Estimate environment temperature using version 1 method with optional smoothing filter.
    
    Uses the 5th percentile method and applies exponential moving average smoothing
    to reduce noise and provide stable estimates.
    
    Args:
        temp_array: Temperature array (96x96, uint16 raw values or float32 Celsius)
        min_temp: Minimum temperature in Celsius from metadata
        max_temp: Maximum temperature in Celsius from metadata
        smoothing: Whether to apply smoothing filter (default: True)
        smoothing_factor: Smoothing factor for exponential moving average (default: 0.3)
                         Lower values (0.1-0.3) = more smoothing, slower response
                         Higher values (0.4-0.7) = less smoothing, faster response
    
    Returns:
        Estimated environment temperature in Celsius (5th percentile, smoothed), or None if input is invalid
    """
    global _env_temp_smoothed
    
    # Get raw estimate
    raw_estimate = estimate_environment_temperature(temp_array, min_temp, max_temp, percentile=5.0)
    
    if raw_estimate is None:
        return None
    
    # If smoothing is disabled, return raw estimate
    if not smoothing:
        return raw_estimate
    
    # Apply exponential moving average smoothing
    if _env_temp_smoothed is None:
        # First estimate - initialize with raw value
        _env_temp_smoothed = raw_estimate
    else:
        # Apply exponential moving average: smoothed = α * new + (1-α) * old
        _env_temp_smoothed = smoothing_factor * raw_estimate + (1 - smoothing_factor) * _env_temp_smoothed
    
    # Also maintain a history for additional smoothing options
    _env_temp_history.append(raw_estimate)
    if len(_env_temp_history) > _MAX_HISTORY_SIZE:
        _env_temp_history.pop(0)
    
    return _env_temp_smoothed


def reset_environment_temperature_smoothing():
    """
    Reset the smoothing filter state.
    
    Call this when starting a new session or when you want to clear the smoothing history.
    """
    global _env_temp_history, _env_temp_smoothed
    _env_temp_history = []
    _env_temp_smoothed = None


def get_environment_temperature_history() -> list:
    """
    Get the history of raw environment temperature estimates.
    
    Returns:
        List of recent raw temperature estimates
    """
    return _env_temp_history.copy()


def estimate_body_temperature(
    environment_temp: float,
    alpha: float = 0.6,
    core_temp: float = 37.0
) -> float:
    """
    Estimate body (skin) temperature from environment temperature.
    
    Uses the formula: Ts = Te + α × (Tc − Te)
    
    Where:
    - Ts = Skin temperature (estimated body temperature)
    - Te = Environment temperature
    - Tc = Core body temperature (default: 37°C)
    - α = Blood flow regulation coefficient
    
    Args:
        environment_temp: Environment/room temperature in Celsius
        alpha: Blood flow regulation coefficient (default: 0.6)
               - Face/torso: α ≈ 0.5–0.7 (default: 0.6)
               - Hands/feet: α ≈ 0.2–0.4
        core_temp: Core body temperature in Celsius (default: 37.0)
    
    Returns:
        Estimated body (skin) temperature in Celsius
    
    Examples:
        >>> # Face/torso at 10°C room temperature
        >>> estimate_body_temperature(10.0, alpha=0.6)
        26.2  # ≈ 10 + 0.6 × (37 - 10) = 26.2°C
        
        >>> # Hands/feet at 10°C room temperature
        >>> estimate_body_temperature(10.0, alpha=0.25)
        16.75  # ≈ 10 + 0.25 × (37 - 10) = 16.75°C
    """
    return environment_temp + alpha * (core_temp - environment_temp)


def estimate_body_temperature_range(
    environment_temp: float,
    core_temp: float = 37.0
) -> tuple[float, float]:
    """
    Estimate body temperature range for different body parts.
    
    Returns temperature ranges for:
    - Face/torso (α ≈ 0.5–0.7)
    - Hands/feet (α ≈ 0.2–0.4)
    
    Args:
        environment_temp: Environment/room temperature in Celsius
        core_temp: Core body temperature in Celsius (default: 37.0)
    
    Returns:
        Tuple of (min_temp, max_temp) where:
        - min_temp: Estimated temperature for hands/feet (α=0.2)
        - max_temp: Estimated temperature for face/torso (α=0.7)
    """
    min_temp = estimate_body_temperature(environment_temp, alpha=0.2, core_temp=core_temp)
    max_temp = estimate_body_temperature(environment_temp, alpha=0.7, core_temp=core_temp)
    return (min_temp, max_temp)

