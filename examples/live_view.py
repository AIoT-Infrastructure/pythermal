#!/usr/bin/env python3
"""
Example: Live view with color/temperature display switching

This example demonstrates a live thermal camera view with:
- Toggle between YUYV grayscale and temperature colorized view
- Mouse hover to see temperature at cursor
- Keyboard controls for switching views
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

from pythermal import ThermalDevice, ThermalLiveView, ThermalSharedMemory, WIDTH, HEIGHT


class EnhancedLiveView(ThermalLiveView):
    """Enhanced live view with additional color modes"""
    
    def __init__(self, device: Optional[ThermalDevice] = None):
        super().__init__(device)
        # Extended view modes
        self.view_modes = ['yuyv', 'temperature', 'temperature_celsius']
        self.view_mode_index = 0
        self.view_mode = self.view_modes[self.view_mode_index]
        
    def get_temperature_celsius_view(self, temp_array: np.ndarray, 
                                     min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to colorized view using actual Celsius values
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature from metadata
            max_temp: Maximum temperature from metadata
            
        Returns:
            BGR image (240x240) with temperature data colorized
        """
        # Get temperature map in Celsius
        temp_celsius = self.shm_reader.get_temperature_map_celsius()
        if temp_celsius is None:
            # Fallback to raw temperature array
            temp_float = temp_array.astype(np.float32)
            raw_min = np.min(temp_float)
            raw_max = np.max(temp_float)
            raw_range = raw_max - raw_min
            if raw_range > 0:
                normalized = (temp_float - raw_min) / raw_range
                temp_celsius = min_temp + normalized * (max_temp - min_temp)
            else:
                temp_celsius = np.full((96, 96), (min_temp + max_temp) / 2.0, dtype=np.float32)
        
        # Upscale from 96x96 to 240x240
        upscaled = cv2.resize(temp_celsius, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255 range
        temp_range = max_temp - min_temp
        if temp_range > 0:
            normalized = ((upscaled - min_temp) / temp_range) * 255.0
            normalized = normalized.clip(0, 255).astype(np.uint8)
        else:
            normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        
        # Apply colormap (using COLORMAP_JET for better visualization)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored
    
    def toggle_view_mode(self):
        """Toggle to next view mode"""
        self.view_mode_index = (self.view_mode_index + 1) % len(self.view_modes)
        self.view_mode = self.view_modes[self.view_mode_index]
        print(f"Switched to {self.view_mode.upper()} view")
    
    def run(self):
        """Main loop for live view"""
        if not self.initialize_shared_memory():
            return
        
        window_name = "PyThermal Live View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 640)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  't' - Toggle view mode (YUYV / Temperature / Temperature Celsius)")
        print("  Mouse - Hover over image to see temperature")
        print()
        
        try:
            while True:
                # Check for new frame
                if not self.shm_reader.has_new_frame():
                    time.sleep(0.01)
                    continue
                
                # Get metadata
                metadata = self.shm_reader.get_metadata()
                if not metadata:
                    time.sleep(0.01)
                    continue
                
                seq_val = metadata.seq
                min_temp = metadata.min_temp
                max_temp = metadata.max_temp
                avg_temp = metadata.avg_temp
                
                # Read and display based on view mode
                if self.view_mode == 'yuyv':
                    yuyv = self.shm_reader.get_yuyv_frame()
                    if yuyv is None:
                        time.sleep(0.01)
                        continue
                    
                    temp_array = self.shm_reader.get_temperature_array()
                    self.current_temp_data = temp_array.copy() if temp_array is not None else None
                    thermal_image = self.get_original_yuyv(yuyv)
                
                elif self.view_mode == 'temperature':
                    temp_array = self.shm_reader.get_temperature_array()
                    if temp_array is not None:
                        self.current_temp_data = temp_array.copy()
                        thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                    else:
                        yuyv = self.shm_reader.get_yuyv_frame()
                        if yuyv is not None:
                            thermal_image = self.get_original_yuyv(yuyv)
                        else:
                            time.sleep(0.01)
                            continue
                
                elif self.view_mode == 'temperature_celsius':
                    temp_array = self.shm_reader.get_temperature_array()
                    if temp_array is not None:
                        self.current_temp_data = temp_array.copy()
                        thermal_image = self.get_temperature_celsius_view(temp_array, min_temp, max_temp)
                    else:
                        yuyv = self.shm_reader.get_yuyv_frame()
                        if yuyv is not None:
                            thermal_image = self.get_original_yuyv(yuyv)
                        else:
                            time.sleep(0.01)
                            continue
                
                else:
                    time.sleep(0.01)
                    continue
                
                # Calculate FPS
                self.frame_count += 1
                fps = self.calculate_fps()
                
                # Draw overlay with mouse temperature
                thermal_image = self.draw_overlay(
                    thermal_image, min_temp, max_temp, avg_temp, seq_val, fps
                )
                
                # Display image
                cv2.imshow(window_name, thermal_image)
                
                # Mark frame as read
                self.shm_reader.mark_frame_read()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.toggle_view_mode()
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nStopping live view...")
        except Exception as e:
            print(f"Error during live view: {e}")
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Live thermal camera view with color/temperature display switching"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional path to native directory (default: use package location)"
    )
    
    args = parser.parse_args()
    
    # Create device if custom path provided
    device = None
    if args.device:
        device = ThermalDevice(native_dir=args.device)
    
    viewer = EnhancedLiveView(device=device)
    viewer.run()


if __name__ == "__main__":
    main()

