#!/usr/bin/env python3
"""
Live View for Thermal Camera - Original Grayscale with Mouse Temperature

Displays real-time thermal imaging feed in original grayscale.
Shows temperature at mouse cursor position.
Press 't' to toggle between YUYV view and 96x96 temperature view.
"""

import numpy as np
import cv2
import time
from datetime import datetime
from typing import Optional

from .thermal_shared_memory import (
    ThermalSharedMemory,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT
)
from .device import ThermalDevice


class ThermalLiveView:
    """Live view display for thermal camera - original grayscale only"""
    
    def __init__(self, device: Optional[ThermalDevice] = None):
        """
        Initialize thermal live view.
        
        Args:
            device: Optional ThermalDevice instance. If None, creates a new one.
        """
        self.device = device
        self._device_owned = device is None  # Track if we created the device
        self.shm_reader: Optional[ThermalSharedMemory] = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # View mode: 'yuyv' or 'temperature'
        self.view_mode = 'yuyv'
        
        # Mouse position tracking for temperature display
        self.mouse_x = -1
        self.mouse_y = -1
        self.current_temp_data = None
        
    def initialize_shared_memory(self) -> bool:
        """Initialize shared memory connection"""
        # Create device if not provided
        if self.device is None:
            self.device = ThermalDevice()
        
        # Start device if not running
        if not self.device.is_running():
            if not self.device.start():
                print("Error: Failed to start thermal device")
                print("Make sure the thermal camera is connected and permissions are set up.")
                return False
        
        # Get shared memory reader
        self.shm_reader = self.device.get_shared_memory()
        
        # Read initial metadata to verify connection
        metadata = self.shm_reader.get_metadata()
        if metadata:
            print(f"Shared memory initialized: {metadata.width}x{metadata.height}")
            print("Press 'q' to quit")
            print("Press 't' to toggle between YUYV view and 96x96 temperature view")
            print("Move mouse over image to see temperature at cursor")
            return True
        else:
            print("Failed to read metadata from shared memory")
            return False
    
    def calculate_temperature_from_pixel(self, x: int, y: int, min_temp: float, max_temp: float) -> Optional[float]:
        """
        Calculate temperature at pixel position using raw temperature data from buffer
        
        Args:
            x: X coordinate (column, 0-239)
            y: Y coordinate (row, 0-239)
            min_temp: Minimum temperature from metadata
            max_temp: Maximum temperature from metadata
            
        Returns:
            Temperature in Celsius, or None if invalid position or data unavailable
        """
        if self.current_temp_data is None:
            return None
        
        # Check bounds
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return None
        
        # Map 240x240 coordinates to 96x96
        temp_x = int((x / WIDTH) * TEMP_WIDTH)
        temp_y = int((y / HEIGHT) * TEMP_HEIGHT)
        
        # Clamp to valid range
        temp_x = max(0, min(temp_x, TEMP_WIDTH - 1))
        temp_y = max(0, min(temp_y, TEMP_HEIGHT - 1))
        
        # Get raw temperature value (16-bit integer)
        raw_temp = self.current_temp_data[temp_y, temp_x]
        
        # Convert raw value to Celsius
        # Map the raw value range to the temperature range from metadata
        raw_min = np.min(self.current_temp_data)
        raw_max = np.max(self.current_temp_data)
        raw_range = raw_max - raw_min
        
        if raw_range > 0:
            # Normalize raw value to 0-1 range, then map to temperature range
            normalized = (raw_temp - raw_min) / raw_range
            temperature = min_temp + normalized * (max_temp - min_temp)
            return temperature
        else:
            # All values are the same, return the metadata average
            return (min_temp + max_temp) / 2.0
    
    def get_original_yuyv(self, yuyv_data: np.ndarray) -> np.ndarray:
        """
        Convert YUYV to grayscale BGR for original view
        
        Args:
            yuyv_data: YUYV frame data
            
        Returns:
            Grayscale BGR image
        """
        # Extract Y channel (grayscale)
        y_channel = yuyv_data[:, :, 0]
        
        # Convert to BGR (3 channels, same grayscale values)
        bgr = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        
        return bgr
    
    def get_temperature_view(self, temp_array: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to visualizable BGR image
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature for normalization
            max_temp: Maximum temperature for normalization
            
        Returns:
            BGR image (240x240) with temperature data upscaled and colorized
        """
        # Convert uint16 to float32 for processing
        temp_float = temp_array.astype(np.float32)
        
        # Get actual min/max from the array
        raw_min = np.min(temp_float)
        raw_max = np.max(temp_float)
        raw_range = raw_max - raw_min
        
        # Normalize to 0-255 range
        # Map raw values to temperature range, then normalize to 0-255
        temp_range = max_temp - min_temp
        if temp_range > 0 and raw_range > 0:
            # First map raw values to temperature values
            # Then normalize temperature values to 0-255
            normalized = ((temp_float - raw_min) / raw_range) * 255.0
            normalized = normalized.clip(0, 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(temp_array, dtype=np.uint8)
        
        # Upscale from 96x96 to 240x240 using INTER_LINEAR
        upscaled = cv2.resize(normalized, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap for better visualization (using COLORMAP_HOT)
        colored = cv2.applyColorMap(upscaled, cv2.COLORMAP_HOT)
        
        return colored
    
    def draw_overlay(self, image: np.ndarray, min_temp: float, max_temp: float, avg_temp: float, 
                     seq: int, fps: float) -> np.ndarray:
        """
        Draw temperature and statistics overlay below the frame
        
        Args:
            image: BGR image to draw on (240x240)
            min_temp: Minimum temperature
            max_temp: Maximum temperature
            avg_temp: Average temperature
            seq: Frame sequence number
            fps: Current FPS
            
        Returns:
            Extended image with text overlay below frame
        """
        # Create extended canvas: 240x240 thermal image + 80px text area below
        text_area_height = 80
        extended_height = HEIGHT + text_area_height
        extended_image = np.zeros((extended_height, WIDTH, 3), dtype=np.uint8)
        
        # Place thermal image at the top
        extended_image[0:HEIGHT, 0:WIDTH] = image
        
        # Draw text area background below the frame
        text_area_y_start = HEIGHT
        cv2.rectangle(extended_image, (0, text_area_y_start), (WIDTH, extended_height), (0, 0, 0), -1)
        
        # Draw temperature information below the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        
        y_offset = text_area_y_start + 15
        line_height = 15
        
        # View mode indicator
        view_text = f"View: {self.view_mode.upper()}"
        cv2.putText(extended_image, view_text, (5, y_offset), font, font_scale, (0, 255, 0), thickness)
        y_offset += line_height
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(extended_image, f"Time: {timestamp}", (5, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Temperature readings
        cv2.putText(extended_image, f"Min: {min_temp:.1f}C", (5, y_offset), font, font_scale, (100, 255, 255), thickness)
        y_offset += line_height
        cv2.putText(extended_image, f"Max: {max_temp:.1f}C", (5, y_offset), font, font_scale, (0, 100, 255), thickness)
        y_offset += line_height
        cv2.putText(extended_image, f"Avg: {avg_temp:.1f}C", (5, y_offset), font, font_scale, (100, 200, 255), thickness)
        y_offset += line_height
        
        # Frame info
        cv2.putText(extended_image, f"Seq: {seq} | FPS: {fps:.1f}", (5, y_offset), font, font_scale, color, thickness)
        
        # Draw mouse temperature if mouse is over image (only in thermal frame area)
        if self.mouse_x >= 0 and self.mouse_y >= 0:
            temp_at_mouse = self.calculate_temperature_from_pixel(
                self.mouse_x, self.mouse_y, min_temp, max_temp
            )
            
            if temp_at_mouse is not None:
                # Draw temperature near cursor (in thermal image area)
                temp_text = f"{temp_at_mouse:.1f}°C"
                text_size = cv2.getTextSize(temp_text, font, font_scale, thickness)[0]
                
                # Position text above cursor, or below if too close to top
                text_x = self.mouse_x - text_size[0] // 2
                text_y = self.mouse_y - 10 if self.mouse_y > 30 else self.mouse_y + 20
                
                # Ensure text stays within thermal image bounds
                text_x = max(5, min(text_x, WIDTH - text_size[0] - 5))
                
                # Draw background rectangle for better visibility
                cv2.rectangle(extended_image, 
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # Draw temperature text
                cv2.putText(extended_image, temp_text, (text_x, text_y), 
                           font, font_scale, (0, 255, 255), thickness)
                
                # Draw crosshair at mouse position
                cv2.drawMarker(extended_image, (self.mouse_x, self.mouse_y), 
                             (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
        
        return extended_image
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            self.frame_count = 0
            self.last_fps_time = current_time
        
        return self.fps
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to track mouse position - uses x/y from window coordinates"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Window is displayed at 2x scale (480x640 for 240x320 image)
            # Mouse coordinates (x, y) are relative to the displayed image size
            # OpenCV mouse coordinates are relative to the image, not the window
            # So we can use them directly, but need to check bounds
            
            # Check if mouse is within thermal image bounds (0 to 239 for 240x240 thermal area)
            # Note: y coordinate should only be tracked in the thermal image area, not the text area below
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                self.mouse_x = x
                self.mouse_y = y
            else:
                # Mouse outside thermal image bounds (could be in text area or outside)
                self.mouse_x = -1
                self.mouse_y = -1
    
    def run(self):
        """Main loop for live view"""
        if not self.initialize_shared_memory():
            return
        
        window_name = "Thermal Camera Live View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Extended image: 240x320 (240 thermal + 80 text area), displayed at 2x scale
        cv2.resizeWindow(window_name, 480, 640)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
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
                    # Read YUYV data
                    yuyv = self.shm_reader.get_yuyv_frame()
                    if yuyv is None:
                        time.sleep(0.01)
                        continue
                    
                    # Read temperature data for accurate temperature calculation
                    temp_array = self.shm_reader.get_temperature_array()
                    self.current_temp_data = temp_array.copy() if temp_array is not None else None
                    
                    # Show original grayscale
                    thermal_image = self.get_original_yuyv(yuyv)
                else:  # temperature view
                    # Read temperature array
                    temp_array = self.shm_reader.get_temperature_array()
                    if temp_array is not None:
                        self.current_temp_data = temp_array.copy()
                        
                        # Show temperature view
                        thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                    else:
                        # Fallback to YUYV if temperature data unavailable
                        yuyv = self.shm_reader.get_yuyv_frame()
                        if yuyv is not None:
                            thermal_image = self.get_original_yuyv(yuyv)
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
                    # Toggle view mode
                    self.view_mode = 'temperature' if self.view_mode == 'yuyv' else 'yuyv'
                    print(f"Switched to {self.view_mode.upper()} view")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping live view...")
        except Exception as e:
            print(f"Error during live view: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        # Cleanup device only if we created it
        if self.device is not None and self._device_owned:
            self.device.cleanup()
        cv2.destroyAllWindows()
        print("Live view stopped")


def main():
    """Main entry point"""
    viewer = ThermalLiveView()
    viewer.run()


if __name__ == "__main__":
    main()
