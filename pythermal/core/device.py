#!/usr/bin/env python3
"""
Thermal Device Manager

Manages the thermal camera device initialization by starting pythermal-recorder
in a separate process and providing access to thermal data via shared memory.
"""

import os
import platform
import subprocess
import time
import signal
import atexit
from pathlib import Path
from typing import Optional
from .thermal_shared_memory import ThermalSharedMemory


def _detect_native_directory() -> str:
    """
    Detect the system architecture and return the appropriate native directory name.
    
    Returns:
        Directory name: 'linux64' for x86_64/amd64, 'armLinux' for ARM architectures
    """
    machine = platform.machine().lower()
    
    # Check for x86_64 architectures
    if machine in ('x86_64', 'amd64'):
        return 'linux64'
    
    # Check for ARM architectures
    if machine in ('arm64', 'aarch64', 'armhf', 'armv7l', 'armv6l'):
        return 'armLinux'
    
    # Default to armLinux for unknown architectures (backward compatibility)
    return 'armLinux'


class ThermalDevice:
    """
    Manages thermal camera device initialization and shared memory access.
    
    This class starts the pythermal-recorder process in the background and
    provides access to thermal data through the shared memory interface.
    """
    
    def __init__(self, native_dir: Optional[str] = None, device_index: Optional[int] = None):
        """
        Initialize thermal device manager.
        
        Args:
            native_dir: Optional path to native directory containing pythermal-recorder.
                       If None, uses default package location.
            device_index: Index of the USB device to use (0 for first device, 1 for second, etc.).
                         If None, uses the smallest available device ID from the mapping file.
                         Default is None. Each device uses a separate shared memory segment.
        """
        if native_dir is None:
            # Default to package location - detect architecture automatically
            package_dir = Path(__file__).parent.parent
            arch_dir = _detect_native_directory()
            self.native_dir = package_dir / "_native" / arch_dir
        else:
            self.native_dir = Path(native_dir)
        
        # Handle device_index: if None, enumerate and use smallest available
        from .device_manager import DeviceManager
        self._device_manager = DeviceManager()
        
        if device_index is None:
            # Enumerate devices to find available ones
            devices = self._device_manager.enumerate_devices_via_sdk(self.native_dir)
            
            if devices:
                # Get device_ids from mapping (consistent IDs based on serial numbers)
                device_ids = []
                enum_to_device_map = {}  # enum_index -> device_id
                
                for device in devices:
                    serial = device.get('serial_number', '')
                    enum_idx = device.get('enum_index', -1)
                    if serial and enum_idx >= 0:
                        device_id = self._device_manager.get_device_id(serial)
                        device_ids.append(device_id)
                        enum_to_device_map[enum_idx] = device_id
                
                if device_ids:
                    # Use smallest device_id
                    device_index = min(device_ids)
                    # Find enum_index for this device_id
                    enum_index = None
                    for eidx, did in enum_to_device_map.items():
                        if did == device_index:
                            enum_index = eidx
                            break
                    if enum_index is None:
                        enum_index = min(enum_to_device_map.keys()) if enum_to_device_map else 0
                else:
                    # No serial numbers, use smallest enum_index
                    enum_indices = [d.get('enum_index', 0) for d in devices if d.get('enum_index', -1) >= 0]
                    enum_index = min(enum_indices) if enum_indices else 0
                    device_index = enum_index  # Use enum_index as device_id initially
            else:
                # No devices enumerated, try smallest from mapping
                device_index = self._device_manager.get_smallest_available_device_id()
                if device_index is None:
                    # No devices in mapping yet, default to 0
                    device_index = 0
                enum_index = device_index  # Use device_index as enum_index initially
        else:
            # device_index provided - map it to enum_index if mapping exists
            devices = self._device_manager.enumerate_devices_via_sdk(self.native_dir)
            enum_index = None
            
            # First, try to find the serial number for the requested device_id
            expected_serial = self._device_manager.get_serial_by_id(device_index)
            
            if expected_serial:
                # We have a serial number for this device_id - find the device with that serial
                if devices:
                    for device in devices:
                        serial = device.get('serial_number', '')
                        enum_idx = device.get('enum_index', -1)
                        if serial == expected_serial and enum_idx >= 0:
                            enum_index = enum_idx
                            self._device_serial = serial
                            break
            else:
                # No serial number mapped for this device_id yet
                # This could be a new device or the mapping hasn't been created yet
                # Try to match by checking if any device's mapped device_id matches
                if devices:
                    for device in devices:
                        serial = device.get('serial_number', '')
                        enum_idx = device.get('enum_index', -1)
                        if serial and enum_idx >= 0:
                            mapped_device_id = self._device_manager.get_device_id(serial)
                            if mapped_device_id == device_index:
                                enum_index = enum_idx
                                self._device_serial = serial
                                break
                        elif enum_idx == device_index:
                            # No serial but enum_index matches - use it directly
                            enum_index = device_index
                            break
            
            # If still no match found, check if we have any devices at all
            if enum_index is None:
                if devices:
                    # Device with requested device_id is not currently connected
                    # Use smallest available enum_index as fallback, but keep device_index unchanged
                    enum_indices = [d.get('enum_index', 0) for d in devices if d.get('enum_index', -1) >= 0]
                    if enum_indices:
                        enum_index = min(enum_indices)
                        # Try to get serial for this device and update mapping if needed
                        for device in devices:
                            if device.get('enum_index', -1) == enum_index:
                                serial = device.get('serial_number', '')
                                if serial:
                                    # Update mapping to use the requested device_id for this serial
                                    self._device_manager.update_mapping(serial, device_index)
                                    self._device_serial = serial
                                break
                else:
                    # No devices enumerated - use device_index as enum_index
                    enum_index = device_index
        
        self.device_index = device_index  # Store the device_id (consistent ID)
        self._enum_index = enum_index  # Store the enum_index (for C++ code)
        self._device_serial = None  # Will be set after device login
        
        # Generate shared memory name based on device index
        from .thermal_shared_memory import get_shm_name
        shm_name = get_shm_name(device_index)
        
        self.recorder_path = self.native_dir / "pythermal-recorder"
        self.process: Optional[subprocess.Popen] = None
        self.shm_reader = ThermalSharedMemory(shm_name=shm_name)
        self._is_running = False
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start(self, timeout: float = 10.0) -> bool:
        """
        Start the thermal recorder process and initialize shared memory.
        
        Args:
            timeout: Maximum time to wait for shared memory to become available (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        if self._is_running:
            return True
        
        # Check if recorder executable exists
        if not self.recorder_path.exists():
            raise FileNotFoundError(
                f"pythermal-recorder not found at {self.recorder_path}. "
                "Make sure the native binaries are installed."
            )
        
        if not os.access(self.recorder_path, os.X_OK):
            raise PermissionError(
                f"pythermal-recorder is not executable: {self.recorder_path}"
            )
        
        # Start the recorder process
        try:
            # Change to native directory to ensure proper library loading
            # Pass enum_index (for device selection) and device_index (for shared memory naming)
            # argv[1] = enum_index (SDK's internal index)
            # argv[2] = device_index (consistent ID for shared memory)
            cmd = [str(self.recorder_path), str(self._enum_index), str(self.device_index)]
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.native_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start thermal recorder: {e}")
        
        # Wait for shared memory to become available
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shm_reader.initialize():
                # Verify we can read metadata
                metadata = self.shm_reader.get_metadata()
                if metadata is not None:
                    # Try to get device serial number and update mapping
                    self._update_device_mapping()
                    self._is_running = True
                    return True
            
            # Check if process died
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"Thermal recorder process exited unexpectedly. "
                    f"Return code: {self.process.returncode}\n"
                    f"STDOUT: {stdout.decode() if stdout else 'None'}\n"
                    f"STDERR: {stderr.decode() if stderr else 'None'}"
                )
            
            time.sleep(0.1)
        
        # Timeout - cleanup and raise error
        self.stop()
        raise TimeoutError(
            f"Shared memory did not become available within {timeout} seconds. "
            "Make sure the thermal camera is connected and permissions are set up."
        )
    
    def stop(self):
        """Stop the thermal recorder process and cleanup resources."""
        if self.process is not None:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for process to terminate (with timeout)
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
            except ProcessLookupError:
                # Process already terminated
                pass
            except Exception as e:
                print(f"Warning: Error stopping thermal recorder: {e}")
            finally:
                self.process = None
        
        self.shm_reader.cleanup()
        self._is_running = False
    
    def is_running(self) -> bool:
        """Check if the thermal recorder is running."""
        if not self._is_running:
            return False
        
        # Verify process is still alive
        if self.process is not None and self.process.poll() is not None:
            self._is_running = False
            return False
        
        return True
    
    def _update_device_mapping(self):
        """Update device mapping with serial number from current device."""
        try:
            if self._device_manager is None:
                from .device_manager import DeviceManager
                self._device_manager = DeviceManager()
            
            # Try to enumerate devices to get serial number for current enum_index
            devices = self._device_manager.enumerate_devices_via_sdk(self.native_dir)
            
            # Find device matching current enum_index
            for device in devices:
                enum_idx = device.get('enum_index', -1)
                if enum_idx == self._enum_index:
                    serial = device.get('serial_number', '')
                    if serial:
                        # Update mapping: serial -> device_id (consistent ID)
                        device_id = self._device_manager.get_device_id(serial)
                        self._device_serial = serial
                        # Ensure device_id matches our device_index
                        if device_id != self.device_index:
                            # Update mapping to use current device_index
                            self._device_manager.update_mapping(serial, self.device_index)
                        break
        except Exception:
            # Failed to update mapping, continue anyway
            pass
    
    def get_shared_memory(self) -> ThermalSharedMemory:
        """
        Get the shared memory reader instance.
        
        Returns:
            ThermalSharedMemory instance for reading thermal data
        """
        if not self._is_running:
            raise RuntimeError("Thermal device is not running. Call start() first.")
        
        return self.shm_reader
    
    def cleanup(self):
        """Cleanup resources (called automatically on exit)."""
        self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor."""
        self.cleanup()

