#!/usr/bin/env python3
"""
Device Manager - Manages consistent device ID mapping

Maps USB thermal devices by serial number to consistent device IDs,
similar to how cv2.VideoCapture assigns camera indices.
Stores the mapping in a file for persistence across sessions.
"""

import json
import subprocess
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class DeviceManager:
    """
    Manages device ID mapping based on USB serial numbers.
    
    Creates a persistent mapping file that stores serial numbers and their
    assigned device IDs, ensuring consistent device identification across
    sessions.
    """
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            mapping_file: Path to device mapping file. If None, uses default location.
        """
        if mapping_file is None:
            # Use user's home directory for the mapping file
            home_dir = Path.home()
            config_dir = home_dir / ".pythermal"
            config_dir.mkdir(exist_ok=True)
            mapping_file = config_dir / "device_mapping.json"
        
        self.mapping_file = Path(mapping_file)
        self.mapping: Dict[str, int] = {}  # serial_number -> device_id
        self._load_mapping()
    
    def _load_mapping(self):
        """Load device mapping from file."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load device mapping: {e}")
                self.mapping = {}
        else:
            self.mapping = {}
    
    def _save_mapping(self):
        """Save device mapping to file."""
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save device mapping: {e}")
    
    def enumerate_devices_via_sdk(self, native_dir: Optional[Path] = None) -> List[Dict]:
        """
        Enumerate devices by querying the SDK via a helper script.
        
        This creates a temporary script that uses the SDK to enumerate devices
        and parse their serial numbers.
        
        Args:
            native_dir: Optional path to native directory
            
        Returns:
            List of device info dictionaries
        """
        devices = []
        
        if native_dir is None:
            from .device import _detect_native_directory
            package_dir = Path(__file__).parent.parent
            arch_dir = _detect_native_directory()
            native_dir = package_dir / "_native" / arch_dir
        
        # Check if usb_demo exists (it has enumeration capability)
        usb_demo_path = native_dir.parent.parent / "hikvision-sdk-builder" / "library" / native_dir.name / "usb_demo"
        if not usb_demo_path.exists():
            # Try alternative location
            usb_demo_path = native_dir / "usb_demo"
        
        # Check if enumerate_devices exists
        enum_path = native_dir / "enumerate_devices"
        if not enum_path.exists():
            # Try alternative location
            enum_path = native_dir.parent.parent / "hikvision-sdk-builder" / "library" / native_dir.name / "enumerate_devices"
        
        if not enum_path.exists():
            return devices
        
        try:
            # Run enumerate_devices and parse JSON output
            result = subprocess.run(
                [str(enum_path)],
                cwd=str(native_dir),
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if result.returncode == 0 and result.stdout:
                import json
                devices = json.loads(result.stdout)
                
                # Update mapping with discovered devices
                for device in devices:
                    serial = device.get('serial_number', '')
                    enum_idx = device.get('enum_index', 0)
                    if serial:
                        # Map serial number to consistent device ID
                        device_id = self.get_device_id(serial)
                        device['device_id'] = device_id
                        
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            # Enumeration failed, return empty list
            pass
        
        return devices
    
    def get_device_id(self, serial_number: str) -> int:
        """
        Get device ID for a given serial number.
        If not in mapping, assigns a new ID (smallest available).
        
        Args:
            serial_number: USB device serial number
            
        Returns:
            Device ID (0, 1, 2, ...)
        """
        if not serial_number or serial_number.strip() == "":
            # Empty serial number - assign next available ID
            used_ids = set(self.mapping.values())
            new_id = 0
            while new_id in used_ids:
                new_id += 1
            # Use a placeholder key for empty serial
            placeholder = f"_empty_{new_id}"
            self.mapping[placeholder] = new_id
            self._save_mapping()
            return new_id
        
        if serial_number in self.mapping:
            return self.mapping[serial_number]
        
        # Assign new ID: smallest available
        used_ids = set(self.mapping.values())
        new_id = 0
        while new_id in used_ids:
            new_id += 1
        
        self.mapping[serial_number] = new_id
        self._save_mapping()
        return new_id
    
    def get_serial_by_id(self, device_id: int) -> Optional[str]:
        """
        Get serial number for a given device ID.
        
        Args:
            device_id: Device ID
            
        Returns:
            Serial number if found, None otherwise
        """
        for serial, dev_id in self.mapping.items():
            if dev_id == device_id:
                return serial
        return None
    
    def get_available_device_ids(self) -> List[int]:
        """
        Get list of available device IDs (sorted).
        
        Returns:
            Sorted list of device IDs
        """
        return sorted(set(self.mapping.values()))
    
    def get_smallest_available_device_id(self) -> Optional[int]:
        """
        Get the smallest available device ID.
        
        Returns:
            Smallest device ID, or None if no devices mapped
        """
        available_ids = self.get_available_device_ids()
        if available_ids:
            return available_ids[0]
        return None
    
    def update_mapping(self, serial_number: str, device_id: int):
        """
        Update or add a device mapping.
        
        Args:
            serial_number: USB device serial number
            device_id: Device ID to assign
        """
        self.mapping[serial_number] = device_id
        self._save_mapping()
    
    def remove_device(self, serial_number: str):
        """
        Remove a device from mapping.
        
        Args:
            serial_number: USB device serial number to remove
        """
        if serial_number in self.mapping:
            del self.mapping[serial_number]
            self._save_mapping()
    
    def list_devices(self) -> List[Tuple[str, int]]:
        """
        List all mapped devices.
        
        Returns:
            List of (serial_number, device_id) tuples, sorted by device_id
        """
        return sorted(self.mapping.items(), key=lambda x: x[1])


def get_device_id_by_serial(serial_number: str, mapping_file: Optional[str] = None) -> int:
    """
    Convenience function to get device ID for a serial number.
    
    Args:
        serial_number: USB device serial number
        mapping_file: Optional path to mapping file
        
    Returns:
        Device ID
    """
    manager = DeviceManager(mapping_file)
    return manager.get_device_id(serial_number)


def get_smallest_device_id(mapping_file: Optional[str] = None) -> Optional[int]:
    """
    Get the smallest available device ID.
    
    Args:
        mapping_file: Optional path to mapping file
        
    Returns:
        Smallest device ID, or None if no devices mapped
    """
    manager = DeviceManager(mapping_file)
    return manager.get_smallest_available_device_id()
