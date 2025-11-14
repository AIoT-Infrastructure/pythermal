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

from pythermal import ThermalLiveView, WIDTH, HEIGHT


class EnhancedLiveView(ThermalLiveView):
    """Enhanced live view with additional color modes"""
    
    def __init__(self, source=None, clahe_clip_limit=2.0):
        """
        Initialize enhanced live view
        
        Args:
            source: File path for recorded .tseq file, or 0/None/empty for live camera (default: live camera)
            clahe_clip_limit: CLAHE clip limit for contrast enhancement (default: 2.0, higher = more contrast)
        """
        super().__init__(source)
        # Extended view modes
        self.view_modes = ['yuyv', 'temperature', 'temperature_celsius', 'temperature_clahe']
        self.view_mode_index = 0
        self.view_mode = self.view_modes[self.view_mode_index]
        self._playback_fps = None
        self.clahe_clip_limit = clahe_clip_limit
        
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
        # Convert raw temperature array to Celsius
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
    
    def get_temperature_clahe_view(self, temp_array: np.ndarray, 
                                   min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to colorized view with CLAHE contrast enhancement
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature from metadata
            max_temp: Maximum temperature from metadata
        
        Returns:
            BGR image (240x240) with temperature data colorized and contrast-enhanced using CLAHE
        """
        # Convert raw temperature array to float
        temp_float = temp_array.astype(np.float32)
        
        # Get actual min/max from the array
        raw_min = np.min(temp_float)
        raw_max = np.max(temp_float)
        raw_range = raw_max - raw_min
        
        # Normalize to 0-255 range for CLAHE processing
        if raw_range > 0:
            normalized = ((temp_float - raw_min) / raw_range) * 255.0
            normalized = normalized.clip(0, 255).astype(np.uint8)
        else:
            normalized = np.zeros((96, 96), dtype=np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Create CLAHE object with specified clip limit
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Upscale from 96x96 to 240x240
        upscaled = cv2.resize(enhanced, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap for visualization
        colored = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)
        
        return colored
    
    def toggle_view_mode(self):
        """Toggle to next view mode"""
        self.view_mode_index = (self.view_mode_index + 1) % len(self.view_modes)
        self.view_mode = self.view_modes[self.view_mode_index]
        print(f"Switched to {self.view_mode.upper()} view")
    
    def run(self):
        """Main loop for live view"""
        if not self.initialize():
            return
        
        is_recorded = self.capture.is_recorded
        window_name = "PyThermal Live View"
        if is_recorded:
            window_name += " (Replay)"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 640)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  't' - Toggle view mode (YUYV / Temperature / Temperature Celsius / Temperature CLAHE)")
        print("  Mouse - Hover over image to see temperature")
        if is_recorded:
            print("  SPACE - Pause/Resume")
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
                    
                    # Read and display based on view mode
                    if self.view_mode == 'yuyv':
                        yuyv = self.capture.get_yuyv_frame()
                        if yuyv is None:
                            if is_recorded:
                                break
                            time.sleep(0.01)
                            continue
                        
                        temp_array = self.capture.get_temperature_array()
                        self.current_temp_data = temp_array.copy() if temp_array is not None else None
                        thermal_image = self.get_original_yuyv(yuyv)
                    
                    elif self.view_mode == 'temperature':
                        temp_array = self.capture.get_temperature_array()
                        if temp_array is not None:
                            self.current_temp_data = temp_array.copy()
                            thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                        else:
                            yuyv = self.capture.get_yuyv_frame()
                            if yuyv is not None:
                                thermal_image = self.get_original_yuyv(yuyv)
                            else:
                                if is_recorded:
                                    break
                                time.sleep(0.01)
                                continue
                    
                    elif self.view_mode == 'temperature_celsius':
                        temp_array = self.capture.get_temperature_array()
                        if temp_array is not None:
                            self.current_temp_data = temp_array.copy()
                            thermal_image = self.get_temperature_celsius_view(temp_array, min_temp, max_temp)
                        else:
                            yuyv = self.capture.get_yuyv_frame()
                            if yuyv is not None:
                                thermal_image = self.get_original_yuyv(yuyv)
                            else:
                                if is_recorded:
                                    break
                                time.sleep(0.01)
                                continue
                    
                    elif self.view_mode == 'temperature_clahe':
                        temp_array = self.capture.get_temperature_array()
                        if temp_array is not None:
                            self.current_temp_data = temp_array.copy()
                            thermal_image = self.get_temperature_clahe_view(temp_array, min_temp, max_temp)
                        else:
                            yuyv = self.capture.get_yuyv_frame()
                            if yuyv is not None:
                                thermal_image = self.get_original_yuyv(yuyv)
                            else:
                                if is_recorded:
                                    break
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
                    self.capture.mark_frame_read()
                    
                    # Handle playback timing for recorded data
                    if is_recorded and self._playback_fps:
                        elapsed = time.time() - last_display_time
                        target_delay = 1.0 / self._playback_fps
                        if elapsed < target_delay:
                            time.sleep(target_delay - elapsed)
                        last_display_time = time.time()
                    elif is_recorded:
                        last_display_time = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.toggle_view_mode()
                elif key == ord(' ') and is_recorded:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                
                if not paused:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nStopping live view...")
        except Exception as e:
            print(f"Error during live view: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Live thermal camera view with color/temperature display switching, or replay recorded files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live view (default)
  python live_view.py
  
  # Recorded file replay
  python live_view.py recordings/thermal_20240101.tseq
  
  # Recorded with custom FPS
  python live_view.py file.tseq --fps 30
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
        "--fps",
        type=float,
        default=None,
        help="Target FPS for recorded data playback (default: use original timestamps, live: no limit)"
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        dest="clahe_clip_limit",
        help="CLAHE clip limit for contrast enhancement (default: 2.0, higher = more contrast, range: 1.0-8.0)"
    )
    
    args = parser.parse_args()
    
    # Create viewer with unified interface
    viewer = EnhancedLiveView(source=args.source, clahe_clip_limit=args.clahe_clip_limit)
    
    # Store FPS for use in run() if needed
    viewer._playback_fps = args.fps
    
    viewer.run()


if __name__ == "__main__":
    main()

