#!/usr/bin/env python3
"""
Device Enumerator Helper

Enumerates USB thermal devices and outputs their information in JSON format.
This can be used to build the device mapping file.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional


def enumerate_devices(native_dir: Optional[Path] = None) -> List[Dict]:
    """
    Enumerate USB thermal devices.
    
    This function attempts to enumerate devices by checking shared memory
    or by using a helper executable. Since we don't have direct SDK bindings,
    we'll use a workaround approach.
    
    Args:
        native_dir: Path to native directory
        
    Returns:
        List of device info dictionaries
    """
    devices = []
    
    # Try to enumerate by checking available devices
    # We'll try indices 0-7 and see which ones respond
    max_devices = 8
    
    for idx in range(max_devices):
        # Check if shared memory exists (indicates device might be active)
        from .thermal_shared_memory import get_shm_name
        shm_name = get_shm_name(idx)
        
        if os.path.exists(shm_name):
            # Device might be active, but we need serial number
            # We'll need to query it via SDK or helper executable
            devices.append({
                'index': idx,
                'serial_number': None,  # Will be filled when device is accessed
                'active': True
            })
    
    return devices


def main():
    """Main entry point for device enumeration."""
    devices = enumerate_devices()
    print(json.dumps(devices, indent=2))


if __name__ == "__main__":
    main()

