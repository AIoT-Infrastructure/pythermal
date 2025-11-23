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
from pythermal.core import ThermalFrameProcessor, ThermalFrame
from pythermal.utils import estimate_environment_temperature_v1


class EnhancedLiveView(ThermalLiveView):
    """Enhanced live view with additional color modes"""
    
    def __init__(self, source=None):
        """
        Initialize enhanced live view
        
        Args:
            source: File path for recorded .tseq or .tframe file, or 0/None/empty for live camera (default: live camera)
        """
        # Check if source is a .tframe file
        self._is_tframe = False
        self._tframe_data = None
        
        if source and isinstance(source, str):
            source_path = Path(source)
            if source_path.exists() and source_path.suffix.lower() == '.tframe':
                self._is_tframe = True
                # Load .tframe file
                self._tframe_data = ThermalFrameProcessor.read_tframe(str(source_path))
                if self._tframe_data is None:
                    raise ValueError(f"Failed to load .tframe file: {source}")
                # Don't initialize parent with source for .tframe files
                super().__init__(None)
            else:
                super().__init__(source)
        else:
            super().__init__(source)
        
        # Extended view modes
        self.view_modes = ['yuyv', 'temperature', 'temperature_celsius']
        self.view_mode_index = 0
        self.view_mode = self.view_modes[self.view_mode_index]
        self._playback_fps = None
        # CLAHE contrast enhancement parameter (clipLimit)
        # Higher values = more contrast enhancement (typical range: 1.0-8.0)
        # Default 3.0 provides noticeable enhancement
        self.clahe_contrast = 3.0
        # Store last frame data for re-rendering during pause
        self._last_frame_data = None
        self._last_metadata = None
        self._last_yuyv = None
        # Store environment temperature
        self._env_temp = None
        
        # If .tframe file, load frame data
        if self._is_tframe and self._tframe_data:
            frame = self._tframe_data['frame']
            self._last_yuyv = frame.yuyv.copy()
            self._last_frame_data = frame.temp_array.copy()
            self._last_metadata = frame.metadata
            self.view_mode = self._tframe_data['view_mode']
            # Find view mode index
            if self.view_mode in self.view_modes:
                self.view_mode_index = self.view_modes.index(self.view_mode)
            self.current_temp_data = frame.temp_array.copy()
        
    def apply_clahe_enhancement(self, image: np.ndarray, is_raw_temp: bool = False) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image
        
        Args:
            image: Grayscale image (uint8 normalized or uint16 raw temperature)
            is_raw_temp: If True, image is raw uint16 temperature data
        
        Returns:
            Enhanced grayscale image with improved contrast (uint8)
        """
        # If raw temperature data, normalize to uint8 first for CLAHE
        if is_raw_temp:
            # Normalize raw uint16 to 0-255 range preserving relative differences
            raw_min = np.min(image)
            raw_max = np.max(image)
            raw_range = raw_max - raw_min
            if raw_range > 0:
                normalized = ((image.astype(np.float32) - raw_min) / raw_range * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            normalized = image
        
        # Create CLAHE object with current contrast setting
        # tileGridSize is the number of tiles (not pixel size)
        # For 96x96 images: (8,8) = 8x8 tiles = 12x12 pixels per tile (good balance)
        # For 240x240 images: (8,8) = 8x8 tiles = 30x30 pixels per tile
        # Smaller number of tiles = larger tiles = less local, more global enhancement
        # Larger number of tiles = smaller tiles = more local enhancement
        if normalized.shape[0] >= 96:
            tile_grid_size = 8  # 8x8 tiles for good local contrast
        elif normalized.shape[0] >= 48:
            tile_grid_size = 4
        else:
            tile_grid_size = 2
        clahe = cv2.createCLAHE(clipLimit=self.clahe_contrast, tileGridSize=(tile_grid_size, tile_grid_size))
        # Apply CLAHE enhancement
        enhanced = clahe.apply(normalized)
        return enhanced
    
    def get_temperature_view(self, temp_array: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
        """
        Convert temperature array to visualizable BGR image with CLAHE enhancement
        
        Args:
            temp_array: 96x96 array of 16-bit temperature values
            min_temp: Minimum temperature for normalization
            max_temp: Maximum temperature for normalization
            
        Returns:
            BGR image (240x240) with temperature data upscaled and colorized
        """
        # Apply CLAHE to raw temperature data BEFORE normalization for better contrast enhancement
        # CLAHE already outputs uint8 in 0-255 range, so no need to re-normalize
        enhanced = self.apply_clahe_enhancement(temp_array, is_raw_temp=True)
        
        # Upscale from 96x96 to 240x240 using INTER_LINEAR
        upscaled = cv2.resize(enhanced, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap for better visualization (using COLORMAP_HOT)
        colored = cv2.applyColorMap(upscaled, cv2.COLORMAP_HOT)
        
        return colored
    
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
        # Apply CLAHE to raw temperature data BEFORE conversion to Celsius
        enhanced = self.apply_clahe_enhancement(temp_array, is_raw_temp=True)
        
        # Convert enhanced temperature array to Celsius
        temp_float = enhanced.astype(np.float32)
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
    
    def adjust_contrast(self, delta: float):
        """Adjust CLAHE contrast level"""
        # Increase range and step size for more noticeable changes
        self.clahe_contrast = max(1.0, min(8.0, self.clahe_contrast + delta))
        print(f"Contrast level: {self.clahe_contrast:.2f} (range: 1.0-8.0)")
    
    def _rerender_frame(self, window_name: str):
        """Re-render the last frame with current settings (used during pause)"""
        if not self._last_metadata:
            return
        
        metadata = self._last_metadata
        seq_val = metadata.seq
        min_temp = metadata.min_temp
        max_temp = metadata.max_temp
        avg_temp = metadata.avg_temp
        
        # Re-render based on current view mode
        if self.view_mode == 'yuyv' and self._last_yuyv is not None:
            thermal_image = self.get_original_yuyv(self._last_yuyv)
        elif self.view_mode == 'temperature' and self._last_frame_data is not None:
            thermal_image = self.get_temperature_view(self._last_frame_data, min_temp, max_temp)
        elif self.view_mode == 'temperature_celsius' and self._last_frame_data is not None:
            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, min_temp, max_temp)
        else:
            return
        
        # Re-estimate environment temperature if we have temperature data
        if self._last_frame_data is not None:
            self._env_temp = estimate_environment_temperature_v1(
                self._last_frame_data, min_temp, max_temp
            )
        
        # Draw overlay
        fps = self.calculate_fps()
        thermal_image = self.draw_overlay(
            thermal_image, min_temp, max_temp, avg_temp, seq_val, fps
        )
        
        # Display image
        cv2.imshow(window_name, thermal_image)
    
    def draw_overlay(self, image: np.ndarray, min_temp: float, max_temp: float, avg_temp: float, 
                     seq: int, fps: float) -> np.ndarray:
        """
        Draw temperature and statistics overlay below the frame with room temperature
        
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
        # Call parent method to get base overlay
        extended_image = super().draw_overlay(image, min_temp, max_temp, avg_temp, seq, fps)
        
        # Add room temperature display if available
        if self._env_temp is not None:
            text_area_height = 80
            text_area_y_start = HEIGHT
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Calculate position for room temperature (right side of temperature readings)
            y_offset = text_area_y_start + 15 + (15 * 3)  # After Min, Max, Avg
            
            # Draw room temperature on the right side
            room_text = f"Room: {self._env_temp:.1f}C"
            text_size = cv2.getTextSize(room_text, font, font_scale, thickness)[0]
            text_x = WIDTH - text_size[0] - 5
            
            # Use a distinct color for room temperature (cyan/light blue)
            cv2.putText(extended_image, room_text, (text_x, y_offset), 
                       font, font_scale, (200, 255, 255), thickness)
        
        return extended_image
    
    def screenshot(self, output_path: Optional[str] = None) -> bool:
        """
        Save current frame as .tframe file (includes rendered image + raw data for replay)
        
        Args:
            output_path: Optional path for .tframe file. If None, generates timestamped filename.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._last_metadata:
            print("No frame data available for screenshot")
            return False
        
        metadata = self._last_metadata
        seq_val = metadata.seq
        min_temp = metadata.min_temp
        max_temp = metadata.max_temp
        avg_temp = metadata.avg_temp
        
        # Render current frame with overlay
        if self.view_mode == 'yuyv' and self._last_yuyv is not None:
            thermal_image = self.get_original_yuyv(self._last_yuyv)
        elif self.view_mode == 'temperature' and self._last_frame_data is not None:
            thermal_image = self.get_temperature_view(self._last_frame_data, min_temp, max_temp)
        elif self.view_mode == 'temperature_celsius' and self._last_frame_data is not None:
            thermal_image = self.get_temperature_celsius_view(self._last_frame_data, min_temp, max_temp)
        else:
            print("No frame data available for screenshot")
            return False
        
        # Re-estimate environment temperature if we have temperature data
        if self._last_frame_data is not None:
            self._env_temp = estimate_environment_temperature_v1(
                self._last_frame_data, min_temp, max_temp
            )
        
        # Draw overlay
        fps = self.calculate_fps()
        thermal_image = self.draw_overlay(
            thermal_image, min_temp, max_temp, avg_temp, seq_val, fps
        )
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"screenshot_{timestamp}.tframe"
        
        output_path = Path(output_path)
        
        # Ensure .tframe extension
        if output_path.suffix.lower() != '.tframe':
            output_path = output_path.with_suffix('.tframe')
        
        # Create ThermalFrame from current data
        try:
            import time
            frame = ThermalFrameProcessor.create_frame_from_capture(self.capture, timestamp=time.time())
            
            if frame is None:
                # Fallback: create frame from stored data
                if self._last_yuyv is not None and self._last_frame_data is not None:
                    rgb = cv2.cvtColor(self._last_yuyv, cv2.COLOR_YUV2RGB_YUYV)
                    frame = ThermalFrame(
                        timestamp=time.time(),
                        metadata=metadata,
                        yuyv=self._last_yuyv.copy(),
                        temp_array=self._last_frame_data.copy(),
                        rgb=rgb
                    )
            
            if frame is not None:
                success = ThermalFrameProcessor.write_tframe(
                    str(output_path), thermal_image, frame, view_mode=self.view_mode
                )
                if success:
                    print(f"Screenshot saved as .tframe: {output_path}")
                    return True
                else:
                    print(f"Failed to save .tframe file: {output_path}")
                    return False
            else:
                print("Could not create frame data for saving")
                return False
        
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_tframe(self):
        """Interactive display for .tframe files"""
        if not self._tframe_data:
            print("No .tframe data loaded")
            return
        
        window_name = "PyThermal Frame Viewer (.tframe)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Get initial rendered image and frame data
        frame = self._tframe_data['frame']
        metadata = frame.metadata
        original_view_mode = self._tframe_data['view_mode']
        needs_rerender = False
        
        # Resize window to fit image
        img_height, img_width = self._tframe_data['rendered_image'].shape[:2]
        cv2.resizeWindow(window_name, img_width, img_height)
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n.tframe File Viewer Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle view mode (YUYV / Temperature / Temperature Celsius)")
        print("  '+' - Increase contrast")
        print("  '-' - Decrease contrast")
        print("  's' - Save screenshot")
        print("  Mouse - Hover over image to see temperature")
        print()
        
        try:
            while True:
                # Re-render based on current view mode if changed
                if needs_rerender or self.view_mode != original_view_mode:
                    # Re-render with current view mode
                    min_temp = metadata.min_temp
                    max_temp = metadata.max_temp
                    avg_temp = metadata.avg_temp
                    
                    if self.view_mode == 'yuyv':
                        thermal_image = self.get_original_yuyv(self._last_yuyv)
                    elif self.view_mode == 'temperature':
                        thermal_image = self.get_temperature_view(self._last_frame_data, min_temp, max_temp)
                    elif self.view_mode == 'temperature_celsius':
                        thermal_image = self.get_temperature_celsius_view(self._last_frame_data, min_temp, max_temp)
                    else:
                        thermal_image = self._tframe_data['rendered_image'].copy()
                    
                    # Re-estimate environment temperature
                    if self._last_frame_data is not None:
                        self._env_temp = estimate_environment_temperature_v1(
                            self._last_frame_data, min_temp, max_temp
                        )
                    
                    # Draw overlay
                    fps = 0.0  # No FPS for single frame
                    thermal_image = self.draw_overlay(
                        thermal_image, min_temp, max_temp, avg_temp, metadata.seq, fps
                    )
                    rendered_image = thermal_image
                    needs_rerender = False
                else:
                    # Use original rendered image
                    rendered_image = self._tframe_data['rendered_image'].copy()
                
                # Display image
                cv2.imshow(window_name, rendered_image)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.toggle_view_mode()
                    needs_rerender = True
                elif key == ord('+') or key == ord('='):
                    self.adjust_contrast(1.0)
                    needs_rerender = True
                elif key == ord('-') or key == ord('_'):
                    self.adjust_contrast(-1.0)
                    needs_rerender = True
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"screenshot_{timestamp}.tframe"
                    # Use current rendered image
                    ThermalFrameProcessor.write_tframe(
                        output_path, rendered_image, frame, view_mode=self.view_mode
                    )
                    print(f"Screenshot saved: {output_path}")
        
        except KeyboardInterrupt:
            print("\nStopping frame viewer...")
        except Exception as e:
            print(f"Error during frame viewing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print("Frame viewer stopped")
    
    def run(self):
        """Main loop for live view"""
        # Handle .tframe files separately
        if self._is_tframe and self._tframe_data:
            self._run_tframe()
            return
        
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
        print("  't' - Toggle view mode (YUYV / Temperature / Temperature Celsius)")
        print("  '+' - Increase contrast")
        print("  '-' - Decrease contrast")
        print("  's' - Screenshot (save current frame)")
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
                    
                    # Store metadata for re-rendering during pause
                    self._last_metadata = metadata
                    
                    # Always try to get both YUYV and temperature data for re-rendering during pause
                    yuyv = self.capture.get_yuyv_frame()
                    temp_array = self.capture.get_temperature_array()
                    
                    # Store frame data for re-rendering (always store both when available)
                    if yuyv is not None:
                        self._last_yuyv = yuyv.copy()
                    if temp_array is not None:
                        self._last_frame_data = temp_array.copy()
                        self.current_temp_data = temp_array.copy()
                    
                    # Read and display based on view mode
                    if self.view_mode == 'yuyv':
                        if yuyv is None:
                            if is_recorded:
                                break
                            time.sleep(0.01)
                            continue
                        thermal_image = self.get_original_yuyv(yuyv)
                    
                    elif self.view_mode == 'temperature':
                        if temp_array is not None:
                            thermal_image = self.get_temperature_view(temp_array, min_temp, max_temp)
                        elif yuyv is not None:
                            thermal_image = self.get_original_yuyv(yuyv)
                        else:
                            if is_recorded:
                                break
                            time.sleep(0.01)
                            continue
                    
                    elif self.view_mode == 'temperature_celsius':
                        if temp_array is not None:
                            thermal_image = self.get_temperature_celsius_view(temp_array, min_temp, max_temp)
                        elif yuyv is not None:
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
                    
                    # Estimate environment temperature if we have temperature data
                    if temp_array is not None:
                        self._env_temp = estimate_environment_temperature_v1(
                            temp_array, min_temp, max_temp
                        )
                    else:
                        self._env_temp = None
                    
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
                    # Re-render current frame if we have frame data (works during pause and live view)
                    if self._last_metadata:
                        self._rerender_frame(window_name)
                elif key == ord('+') or key == ord('='):
                    self.adjust_contrast(1.0)  # Increased step size for more noticeable changes
                    # Re-render current frame if we have frame data (works during pause and live view)
                    if self._last_metadata:
                        self._rerender_frame(window_name)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_contrast(-1.0)  # Increased step size for more noticeable changes
                    # Re-render current frame if we have frame data (works during pause and live view)
                    if self._last_metadata:
                        self._rerender_frame(window_name)
                elif key == ord('s'):
                    self.screenshot()
                elif key == ord(' ') and is_recorded:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    # Re-render immediately when pausing to show current state
                    if paused and self._last_metadata:
                        self._rerender_frame(window_name)
                
                # When paused, continuously re-render to update mouse temperature display
                if paused and self._last_metadata:
                    self._rerender_frame(window_name)
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                elif not paused:
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
  
  # View .tframe file (interactive single frame)
  python live_view.py screenshot_20240101.tframe
        """
    )
    parser.add_argument(
        "source",
        type=str,
        nargs="?",
        default=None,
        help="Source: file path for recorded .tseq or .tframe file, or 0/empty/omit for live camera (default: live camera)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for recorded data playback (default: use original timestamps, live: no limit)"
    )
    
    args = parser.parse_args()
    
    # Create viewer with unified interface
    viewer = EnhancedLiveView(source=args.source)
    
    # Store FPS for use in run() if needed
    viewer._playback_fps = args.fps
    
    viewer.run()


if __name__ == "__main__":
    main()

