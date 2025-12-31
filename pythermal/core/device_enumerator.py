#!/usr/bin/env python3
"""
Device Enumerator - Enumerate USB thermal devices

Uses pythermal-recorder or usb_demo to enumerate available devices
and get their serial numbers for consistent ID mapping.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Optional


def enumerate_devices_via_recorder(native_dir: Optional[Path] = None) -> List[Dict]:
    """
    Enumerate devices by trying to connect to each index.
    
    This is a workaround since we don't have direct SDK bindings.
    We'll try to get device info by attempting connections.
    
    Args:
        native_dir: Path to native directory
        
    Returns:
        List of device info dictionaries
    """
    devices = []
    max_devices = 8
    
    if native_dir is None:
        from .device import _detect_native_directory
        package_dir = Path(__file__).parent.parent
        arch_dir = _detect_native_directory()
        native_dir = package_dir / "_native" / arch_dir
    
    recorder_path = native_dir / "pythermal-recorder"
    
    if not recorder_path.exists():
        return devices
    
    # Try each device index
    for idx in range(max_devices):
        try:
            # We can't easily get serial number without SDK bindings
            # So we'll use a different approach: check if device responds
            # by checking shared memory creation
            
            # Actually, we need to parse output from the recorder or use usb_demo
            # For now, we'll mark devices as potentially available
            # The real mapping will happen when devices are actually used
            
            # Check if shared memory exists (indicates device might be active)
            from .thermal_shared_memory import get_shm_name
            shm_name = get_shm_name(idx)
            
            # Note: This doesn't give us serial numbers
            # We'll need to use a helper executable or modify the approach
            pass
            
        except Exception:
            continue
    
    return devices


def create_device_mapping_helper(native_dir: Optional[Path] = None) -> Dict[str, int]:
    """
    Create device mapping by querying available devices.
    
    Since we can't easily enumerate from Python, we'll use a simpler approach:
    - When a device is first used, we'll try to get its serial number
    - Map serial numbers to consistent IDs
    
    For now, returns empty dict - mapping will be built as devices are used.
    
    Args:
        native_dir: Path to native directory
        
    Returns:
        Dictionary mapping serial_number -> device_id
    """
    # This is a placeholder - actual implementation would query devices
    # For now, we'll build the mapping dynamically as devices are used
    return {}

