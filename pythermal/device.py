#!/usr/bin/env python3
"""
Thermal Device Manager

Manages the thermal camera device initialization by starting pythermal-recorder
in a separate process and providing access to thermal data via shared memory.
"""

import os
import subprocess
import time
import signal
import atexit
from pathlib import Path
from typing import Optional
from .thermal_shared_memory import ThermalSharedMemory


class ThermalDevice:
    """
    Manages thermal camera device initialization and shared memory access.
    
    This class starts the pythermal-recorder process in the background and
    provides access to thermal data through the shared memory interface.
    """
    
    def __init__(self, native_dir: Optional[str] = None):
        """
        Initialize thermal device manager.
        
        Args:
            native_dir: Optional path to native directory containing pythermal-recorder.
                       If None, uses default package location.
        """
        if native_dir is None:
            # Default to package location
            package_dir = Path(__file__).parent
            self.native_dir = package_dir / "_native" / "armLinux"
        else:
            self.native_dir = Path(native_dir)
        
        self.recorder_path = self.native_dir / "pythermal-recorder"
        self.process: Optional[subprocess.Popen] = None
        self.shm_reader = ThermalSharedMemory()
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
            self.process = subprocess.Popen(
                [str(self.recorder_path)],
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

