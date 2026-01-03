#!/usr/bin/env python3
"""
Device Manager - Manages consistent device ID mapping

Maps USB thermal devices by serial number to consistent device IDs,
similar to how cv2.VideoCapture assigns camera indices.
Stores the mapping in a file for persistence across sessions.
"""

import json
import csv
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
        # Camera table file: CSV format for easy viewing/editing
        self.cameras_table_file = self.mapping_file.parent / "cameras"
        self.mapping: Dict[str, int] = {}  # serial_number -> device_id
        self._load_mapping()
    
    def _load_mapping(self):
        """Load device mapping from file (JSON) and camera table (CSV)."""
        # Try loading from JSON first (backward compatibility)
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load device mapping: {e}")
                self.mapping = {}
        
        # Also try loading from CSV table (preferred format)
        if self.cameras_table_file.exists():
            try:
                csv_mapping = self._load_cameras_table()
                if csv_mapping:
                    # Merge CSV data, CSV takes precedence
                    self.mapping.update(csv_mapping)
            except Exception as e:
                print(f"Warning: Failed to load cameras table: {e}")
        
        # If no mapping exists, start fresh
        if not self.mapping:
            self.mapping = {}
    
    def _load_cameras_table(self) -> Dict[str, int]:
        """Load camera mappings from CSV table file."""
        mapping = {}
        if not self.cameras_table_file.exists():
            return mapping
        
        try:
            with open(self.cameras_table_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    serial = row.get('serial_number', '').strip()
                    device_index = row.get('device_index', '').strip()
                    if serial and device_index:
                        try:
                            mapping[serial] = int(device_index)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Warning: Failed to parse cameras table: {e}")
        
        return mapping
    
    def _save_cameras_table(self):
        """Save camera mappings to CSV table file."""
        try:
            self.cameras_table_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Sort by device_index for consistent ordering
            sorted_mappings = sorted(self.mapping.items(), key=lambda x: (x[1], x[0]))
            
            with open(self.cameras_table_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['serial_number', 'device_index'])
                # Write data rows (exclude placeholder entries)
                for serial, device_id in sorted_mappings:
                    if not serial.startswith("_empty_"):
                        writer.writerow([serial, device_id])
        except IOError as e:
            print(f"Warning: Failed to save cameras table: {e}")
    
    def _save_mapping(self):
        """Save device mapping to both JSON and CSV table files."""
        # Save JSON (for backward compatibility)
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save device mapping: {e}")
        
        # Save CSV table (human-readable format)
        self._save_cameras_table()
    
    def enumerate_devices_via_sdk(self, native_dir: Optional[Path] = None) -> List[Dict]:
        """
        Enumerate devices by querying the SDK via a helper script.
        
        This creates a temporary script that uses the SDK to enumerate devices
        and parse their serial numbers. Devices are sorted by serial number
        to ensure consistent ordering.
        
        Args:
            native_dir: Optional path to native directory
            
        Returns:
            List of device info dictionaries, sorted by serial number
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
                
                # Separate devices with and without serial numbers
                devices_with_serial = []
                devices_without_serial = []
                
                for device in devices:
                    serial = device.get('serial_number', '').strip()
                    if serial:
                        devices_with_serial.append(device)
                    else:
                        devices_without_serial.append(device)
                
                # Sort devices by serial number to ensure consistent ordering
                devices_with_serial.sort(key=lambda d: d.get('serial_number', '').strip())
                
                # Normalize device IDs based on sorted serial numbers
                # This ensures cameras are always ordered consistently by serial number
                self._normalize_device_ids(devices_with_serial)
                
                # Assign device IDs to devices
                for device in devices_with_serial:
                    serial = device.get('serial_number', '').strip()
                    if serial:
                        device_id = self.mapping.get(serial, 0)
                        device['device_id'] = device_id
                
                # Handle devices without serial numbers (assign IDs after serialized devices)
                for device in devices_without_serial:
                    enum_idx = device.get('enum_index', 0)
                    # Use enum_index as device_id for devices without serial
                    device['device_id'] = enum_idx
                
                # Combine and return devices (with serial first, sorted)
                devices = devices_with_serial + devices_without_serial
                        
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            # Enumeration failed, return empty list
            pass
        
        return devices
    
    def _normalize_device_ids(self, current_devices: List[Dict]):
        """
        Normalize device IDs based on sorted serial numbers.
        
        This ensures that devices are always assigned IDs based on their
        serial number order (alphabetically sorted), regardless of when
        they were first discovered or their enumeration order.
        
        Args:
            current_devices: List of currently connected devices with serial numbers
        """
        # Get all serial numbers from current devices
        current_serials = [d.get('serial_number', '').strip() 
                          for d in current_devices 
                          if d.get('serial_number', '').strip()]
        
        # Get all serial numbers from existing mapping (including disconnected devices)
        existing_serials = set(self.mapping.keys())
        existing_serials = {s for s in existing_serials if not s.startswith("_empty_")}
        
        # Combine and sort all serial numbers
        all_serials = sorted(set(current_serials) | existing_serials)
        
        # Create new mapping based on sorted order
        new_mapping = {}
        for position, serial in enumerate(all_serials):
            new_mapping[serial] = position
        
        # Preserve placeholder entries (devices without serial numbers)
        for key, value in self.mapping.items():
            if key.startswith("_empty_"):
                new_mapping[key] = value
        
        # Update mapping
        self.mapping = new_mapping
        self._save_mapping()
    
    def get_device_id(self, serial_number: str) -> int:
        """
        Get device ID for a given serial number.
        
        Device IDs are assigned based on sorted serial number order.
        The mapping is normalized when devices are enumerated via enumerate_devices_via_sdk(),
        ensuring consistent ordering regardless of enumeration order.
        
        Args:
            serial_number: USB device serial number
            
        Returns:
            Device ID (0, 1, 2, ...) based on sorted serial number order
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
        
        # Return existing mapping if available
        if serial_number in self.mapping:
            return self.mapping[serial_number]
        
        # New device - assign ID based on sorted position
        # Get all serial numbers (existing + new)
        all_serials = set(self.mapping.keys())
        all_serials.add(serial_number)
        
        # Remove placeholder entries
        all_serials = {s for s in all_serials if not s.startswith("_empty_")}
        
        # Sort all serial numbers
        sorted_serials = sorted(all_serials)
        
        # Find position of this serial in sorted list
        position = sorted_serials.index(serial_number)
        
        # Assign this serial to its sorted position
        self.mapping[serial_number] = position
        self._save_mapping()
        
        return position
    
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
    
    def get_cameras_table_path(self) -> Path:
        """
        Get the path to the cameras table file.
        
        Returns:
            Path to the cameras CSV table file
        """
        return self.cameras_table_file
    
    def print_cameras_table(self):
        """
        Print the cameras table in a human-readable format.
        """
        if not self.mapping:
            print("No cameras registered.")
            return
        
        # Sort by device_index
        sorted_mappings = sorted(self.mapping.items(), key=lambda x: (x[1], x[0]))
        
        print("\n" + "=" * 70)
        print("Registered Cameras")
        print("=" * 70)
        print(f"{'Device Index':<15} {'Serial Number':<50}")
        print("-" * 70)
        
        for serial, device_id in sorted_mappings:
            if not serial.startswith("_empty_"):
                print(f"{device_id:<15} {serial:<50}")
        
        print("=" * 70)
        print(f"\nTable file: {self.cameras_table_file}")
        print(f"JSON file: {self.mapping_file}")
        print()


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
