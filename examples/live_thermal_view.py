#!/usr/bin/env python3
"""
Example: Low-resolution privacy-friendly live view

This example demonstrates the native 96x96 thermal sensor resolution with:
- Display at original 96x96 resolution (no upscaling)
- Boundary smoothing for better visualization
- Privacy-friendly low resolution demonstration
- Temperature colorized view with mouse hover
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import cv2
import time
from datetime import datetime
from typing import Optional

from pythermal import ThermalLiveView, TEMP_WIDTH, TEMP_HEIGHT
from pythermal.utils import estimate_environment_temperature_v1


class LowResLiveView(ThermalLiveView):
    """Low-resolution live view using native 96x96 thermal data"""
    
    def __init__(self, source=None, device_index: int = 0, native_dir: Optional[str] = None):
        """
        Initialize low-resolution live view
        
        Args:
            source: File path for recorded .tseq or .tframe file, or 0/None/empty for live camera (default: live camera)
            device_index: Index of the USB device to use (0 for first device, 1 for second, etc.).
                         Default is 0. Only used for live camera.
            native_dir: Optional path to native directory containing pythermal-recorder.
                       If None, uses default package location. Only used for live camera.
        """
        super().__init__(source=source, device_index=device_index, native_dir=native_dir)
        
        # View modes for low-res display
        self.view_modes = ['temperature', 'temperature_celsius']
        self.view_mode_index = 0
        self.view_mode = self.view_modes[self.view_mode_index]
        
        # Smoothing parameters
        self.smoothing_kernel_size = 3  # Gaussian blur kernel size (must be odd)
        self.smoothing_sigma = 1.0  # Gaussian blur sigma
        
        # Store last frame data
        self._last_frame_data = None
        self._last_metadata = None
        
        # Store environment temperature
        self._env_temp = None
    
    def apply_smoothing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to smooth boundaries
        
        Args:
            image: Input image (96x96)
        
        Returns:
            Smoothed image (96x96)
        """
        # Apply Gaussian blur for boundary smoothing
        smoothed = cv2.GaussianBlur(image, (self.smoothing_kernel_size, self.smoothing_kernel_size), self.smoothing_sigma)
        return smoothed
    
    def get_temperature_view(self, temp_array: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to visualizable BGR image at native 96x96 resolution
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature for normalization
            max_temp: Maximum temperature for normalization
            
        Returns:
            BGR image (96x96) with temperature data colorized
        """
        # Normalize temperature data to 0-255 range
        temp_min = np.min(temp_array)
        temp_max = np.max(temp_array)
        temp_range = temp_max - temp_min
        
        if temp_range > 0:
            normalized = ((temp_array.astype(np.float32) - temp_min) / temp_range * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(temp_array, dtype=np.uint8)
        
        # Apply smoothing to smooth boundaries
        smoothed = self.apply_smoothing(normalized)
        
        # Apply colormap for better visualization (using COLORMAP_HOT)
        colored = cv2.applyColorMap(smoothed, cv2.COLORMAP_HOT)
        
        return colored
    
    def get_temperature_celsius_view(self, temp_array: np.ndarray, 
                                     min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to colorized view using actual Celsius values at native resolution
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature from metadata
            max_temp: Maximum temperature from metadata
        
        Returns:
            BGR image (96x96) with temperature data colorized
        """
        # Convert temperature array to Celsius
        temp_float = temp_array.astype(np.float32)
        raw_min = np.min(temp_float)
        raw_max = np.max(temp_float)
        raw_range = raw_max - raw_min
        
        if raw_range > 0:
            normalized = (temp_float - raw_min) / raw_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
        else:
            temp_celsius = np.full((TEMP_HEIGHT, TEMP_WIDTH), (min_temp + max_temp) / 2.0, dtype=np.float32)
        
        # Apply smoothing to smooth boundaries
        smoothed = self.apply_smoothing(temp_celsius)
        
        # Normalize to 0-255 range
        temp_range = max_temp - min_temp
        if temp_range > 0:
            normalized = ((smoothed - min_temp) / temp_range) * 255.0
            normalized = normalized.clip(0, 255).astype(np.uint8)
        else:
            normalized = np.zeros((TEMP_HEIGHT, TEMP_WIDTH), dtype=np.uint8)
        
        # Apply colormap (using COLORMAP_JET for better visualization)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def toggle_view_mode(self):
        """Toggle to next view mode"""
        self.view_mode_index = (self.view_mode_index + 1) % len(self.view_modes)
        self.view_mode = self.view_modes[self.view_mode_index]
        print(f"Switched to {self.view_mode.upper()} view")
    
    def adjust_smoothing(self, delta: float):
        """Adjust smoothing sigma level"""
        self.smoothing_sigma = max(0.5, min(3.0, self.smoothing_sigma + delta))
        print(f"Smoothing sigma: {self.smoothing_sigma:.2f} (range: 0.5-3.0)")
    
    def calculate_temperature_from_pixel(self, x: int, y: int, min_temp: float, max_temp: float) -> Optional[float]:
        """
        Calculate temperature at pixel position in 96x96 coordinate space
        
        Args:
            x: X coordinate (0-95)
            y: Y coordinate (0-95)
            min_temp: Minimum temperature from metadata
            max_temp: Maximum temperature from metadata
            
        Returns:
            Temperature in Celsius, or None if invalid position
        """
        if self.current_temp_data is None:
            return None
        
        # Check bounds for 96x96 resolution
        if x < 0 or x >= TEMP_WIDTH or y < 0 or y >= TEMP_HEIGHT:
            return None
        
        # Get temperature value at pixel
        temp_value = self.current_temp_data[y, x]
        
        # Convert to Celsius using metadata range
        temp_min = np.min(self.current_temp_data)
        temp_max = np.max(self.current_temp_data)
        temp_range = temp_max - temp_min
        
        if temp_range > 0:
            normalized = (temp_value - temp_min) / temp_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
        else:
            temp_celsius = (min_temp + max_temp) / 2.0
        
        return temp_celsius
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for temperature display"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Store mouse position (in 96x96 coordinate space)
            self.mouse_x = max(0, min(x, TEMP_WIDTH - 1))
            self.mouse_y = max(0, min(y, TEMP_HEIGHT - 1))
    
    def draw_overlay(self, image: np.ndarray, min_temp: float, max_temp: float, avg_temp: float, 
                     seq: int, fps: float) -> np.ndarray:
        """
        Return image without any text overlay (clean display)
        
        Args:
            image: BGR image (96x96)
            min_temp: Minimum temperature (unused)
            max_temp: Maximum temperature (unused)
            avg_temp: Average temperature (unused)
            seq: Frame sequence number (unused)
            fps: Current FPS (unused)
            
        Returns:
            Image without any overlays
        """
        return image
    
    def run(self):
        """Main loop for low-resolution live view"""
        if not self.initialize():
            return
        
        is_recorded = self.capture.is_recorded
        window_name = "PyThermal Live View (96x96)"
        if is_recorded:
            window_name += " (Replay)"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Set window size to show 96x96 image (scaled up for visibility, but maintaining aspect ratio)
        cv2.resizeWindow(window_name, TEMP_WIDTH * 4, TEMP_HEIGHT * 4)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nLow-Resolution Privacy-Friendly View Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle view mode (Temperature / Temperature Celsius)")
        print("  '+' - Increase smoothing")
        print("  '-' - Decrease smoothing")
        print("  's' - Screenshot (save current frame)")
        if is_recorded:
            print("  SPACE - Pause/Resume")
        print(f"  Displaying at native {TEMP_WIDTH}x{TEMP_HEIGHT} resolution (no text overlay)")
        print()
        
        paused = False
        last_display_time = time.time()
        
        try:
            while True:
                if not paused:
                    # Check for new frame
                    if not self.capture.has_new_frame():
                        if is_recorded:
                            print(f"\nEnd of file reached. Processed {self.frame_count} frames")
                            break
                        time.sleep(0.01)
                        continue
                    
                    # Get metadata
                    metadata = self.capture.get_metadata()
                    if not metadata:
                        if is_recorded:
                            break
                        time.sleep(0.01)
                        continue
                    
                    seq_val = metadata.seq
                    min_temp = metadata.min_temp
                    max_temp = metadata.max_temp
                    avg_temp = metadata.avg_temp
                    
                    # Store metadata for re-rendering during pause
                    self._last_metadata = metadata
                    
                    # Get temperature data (96x96)
                    temp_array = self.capture.get_temperature_array()
                    
                    if temp_array is None:
                        if is_recorded:
                            break
                        time.sleep(0.01)
                        continue
                    
                    # Store frame data
                    self._last_frame_data = temp_array.copy()
                    self.current_temp_data = temp_array.copy()
                    
                    # Render based on view mode
                    if self.view_mode == 'temperature':
                        thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                    elif self.view_mode == 'temperature_celsius':
                        thermal_image = self.get_temperature_celsius_view(temp_array, min_temp, max_temp)
                    else:
                        thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                    
                    # Calculate FPS
                    self.frame_count += 1
                    fps = self.calculate_fps()
                    
                    # Estimate environment temperature
                    self._env_temp = estimate_environment_temperature_v1(
                        temp_array, min_temp, max_temp
                    )
                    
                    # Draw overlay with mouse temperature
                    thermal_image = self.draw_overlay(
                        thermal_image, min_temp, max_temp, avg_temp, seq_val, fps
                    )
                    
                    # Display image
                    cv2.imshow(window_name, thermal_image)
                    
                    # Mark frame as read
                    self.capture.mark_frame_read()
                    
                    # Handle playback timing for recorded data
                    if is_recorded:
                        elapsed = time.time() - last_display_time
                        target_delay = 1.0 / 25.0  # Default 25 FPS
                        if elapsed < target_delay:
                            time.sleep(target_delay - elapsed)
                        last_display_time = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.toggle_view_mode()
                    # Re-render current frame
                    if self._last_metadata and self._last_frame_data is not None:
                        metadata = self._last_metadata
                        if self.view_mode == 'temperature':
                            thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        else:
                            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        fps = self.calculate_fps()
                        thermal_image = self.draw_overlay(
                            thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                        )
                        cv2.imshow(window_name, thermal_image)
                elif key == ord('+') or key == ord('='):
                    self.adjust_smoothing(0.2)
                    # Re-render current frame
                    if self._last_metadata and self._last_frame_data is not None:
                        metadata = self._last_metadata
                        if self.view_mode == 'temperature':
                            thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        else:
                            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        fps = self.calculate_fps()
                        thermal_image = self.draw_overlay(
                            thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                        )
                        cv2.imshow(window_name, thermal_image)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_smoothing(-0.2)
                    # Re-render current frame
                    if self._last_metadata and self._last_frame_data is not None:
                        metadata = self._last_metadata
                        if self.view_mode == 'temperature':
                            thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        else:
                            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        fps = self.calculate_fps()
                        thermal_image = self.draw_overlay(
                            thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                        )
                        cv2.imshow(window_name, thermal_image)
                elif key == ord('s'):
                    # Save screenshot
                    if self._last_metadata and self._last_frame_data is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"screenshot_lowres_{timestamp}.png"
                        metadata = self._last_metadata
                        if self.view_mode == 'temperature':
                            thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        else:
                            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        fps = self.calculate_fps()
                        thermal_image = self.draw_overlay(
                            thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                        )
                        cv2.imwrite(output_path, thermal_image)
                        print(f"Screenshot saved: {output_path}")
                elif key == ord(' ') and is_recorded:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    # Re-render immediately when pausing
                    if paused and self._last_metadata and self._last_frame_data is not None:
                        metadata = self._last_metadata
                        if self.view_mode == 'temperature':
                            thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        else:
                            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                        fps = self.calculate_fps()
                        thermal_image = self.draw_overlay(
                            thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                        )
                        cv2.imshow(window_name, thermal_image)
                
                # When paused, continuously re-render to update mouse temperature display
                if paused and self._last_metadata and self._last_frame_data is not None:
                    metadata = self._last_metadata
                    if self.view_mode == 'temperature':
                        thermal_image = self.get_temperature_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                    else:
                        thermal_image = self.get_temperature_celsius_view(self._last_frame_data, metadata.min_temp, metadata.max_temp)
                    fps = self.calculate_fps()
                    thermal_image = self.draw_overlay(
                        thermal_image, metadata.min_temp, metadata.max_temp, metadata.avg_temp, metadata.seq, fps
                    )
                    cv2.imshow(window_name, thermal_image)
                    time.sleep(0.01)
                elif not paused:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nStopping low-res view...")
        except Exception as e:
            print(f"Error during low-res view: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Low-resolution privacy-friendly thermal camera view at native 96x96 resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live view (default)
  python live_thermal_view.py
  
  # Recorded file replay
  python live_thermal_view.py recordings/thermal_20240101.tseq
  
  # Use specific device
  python live_thermal_view.py --device-index 0
        """
    )
    parser.add_argument(
        "source",
        type=str,
        nargs="?",
        default=None,
        help="Source: file path for recorded .tseq file, or 0/empty/omit for live camera (default: live camera)"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index of the USB device to use (0 for first device, 1 for second, etc.). Default: 0"
    )
    
    args = parser.parse_args()
    
    # Create low-res viewer
    viewer = LowResLiveView(source=args.source, device_index=args.device_index)
    viewer.run()


if __name__ == "__main__":
    main()

