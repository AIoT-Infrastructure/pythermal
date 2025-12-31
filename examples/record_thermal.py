#!/usr/bin/env python3
"""
Example: Record thermal data with various formats

This example demonstrates recording thermal camera data with:
- Configurable recording duration
- Multiple output formats: MP4 video, raw temperature arrays, or both
- Temperature data saved to NPY files
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np
import cv2
from datetime import datetime

from pythermal import ThermalDevice, ThermalRecorder, ThermalSharedMemory


class EnhancedRecorder:
    """Enhanced recorder with MP4 and NPY support"""
    
    def __init__(self, output_dir: str = "recordings", device_index: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_index = device_index
        self.device = None
        self.temp_arrays = []  # Store temperature arrays for NPY
        
    def record(self, duration: float, format_type: str = "both", fps: float = 25.0):
        """
        Record thermal data
        
        Args:
            duration: Recording duration in seconds
            format_type: "mp4", "raw", or "both"
            fps: Frames per second for MP4 video
        """
        print(f"Starting recording for {duration} seconds...")
        print(f"Format: {format_type}")
        print(f"FPS: {fps}")
        
        # Initialize device
        self.device = ThermalDevice(device_index=self.device_index)
        if not self.device.start():
            print("Error: Failed to start thermal device")
            return False
        
        shm = self.device.get_shared_memory()
        
        # Create output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.output_dir / f"thermal_{timestamp}"
        
        # Initialize outputs based on format
        video_writer = None
        raw_file = None
        
        if format_type in ["mp4", "both"]:
            video_path = base_name.with_suffix(".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                fps,
                (240, 240),
                True
            )
            print(f"MP4 video: {video_path}")
        
        if format_type in ["raw", "both"]:
            raw_path = base_name.with_suffix(".tseq")
            raw_file = open(raw_path, "wb")
            # Write header
            raw_file.write(b"TSEQ\x01\x01")  # Version 1, color enabled
            print(f"Raw file: {raw_path}")
        
        # NPY file for temperature data
        npy_path = base_name.with_suffix(".npy")
        print(f"Temperature NPY: {npy_path}")
        
        # Recording loop
        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        print("\nRecording... (Press Ctrl+C to stop early)")
        
        try:
            while time.time() < end_time:
                if not shm.has_new_frame():
                    time.sleep(0.01)
                    continue
                
                # Get frame data
                metadata = shm.get_metadata()
                if not metadata:
                    time.sleep(0.01)
                    continue
                
                yuyv = shm.get_yuyv_frame()
                temp_array = shm.get_temperature_array()
                
                if yuyv is None or temp_array is None:
                    time.sleep(0.01)
                    continue
                
                # Convert YUYV to RGB for video
                if format_type in ["mp4", "both"] and video_writer:
                    rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
                    # Convert RGB to BGR for OpenCV
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr)
                
                # Write raw data
                if format_type in ["raw", "both"] and raw_file:
                    import struct
                    timestamp = time.time()
                    frame_header = struct.pack("dIfff", timestamp, metadata.seq,
                                              metadata.min_temp, metadata.max_temp, metadata.avg_temp)
                    raw_file.write(frame_header)
                    raw_file.write(yuyv.tobytes())
                    raw_file.write(temp_array.tobytes())
                    rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
                    raw_file.write(rgb.tobytes())
                
                # Store temperature array for NPY
                self.temp_arrays.append({
                    'timestamp': time.time(),
                    'sequence': metadata.seq,
                    'temperature': temp_array.copy(),
                    'min_temp': metadata.min_temp,
                    'max_temp': metadata.max_temp,
                    'avg_temp': metadata.avg_temp
                })
                
                shm.mark_frame_read()
                frame_count += 1
                
                # Progress update every second
                elapsed = time.time() - start_time
                if frame_count % int(fps) == 0:
                    remaining = duration - elapsed
                    print(f"  Recorded {frame_count} frames ({elapsed:.1f}s / {duration:.1f}s, {remaining:.1f}s remaining)")
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
                print(f"✓ MP4 video saved")
            
            if raw_file:
                raw_file.close()
                print(f"✓ Raw file saved")
            
            # Save temperature arrays to NPY
            if self.temp_arrays:
                temp_data = {
                    'arrays': np.array([item['temperature'] for item in self.temp_arrays]),
                    'timestamps': np.array([item['timestamp'] for item in self.temp_arrays]),
                    'sequences': np.array([item['sequence'] for item in self.temp_arrays]),
                    'min_temps': np.array([item['min_temp'] for item in self.temp_arrays]),
                    'max_temps': np.array([item['max_temp'] for item in self.temp_arrays]),
                    'avg_temps': np.array([item['avg_temp'] for item in self.temp_arrays]),
                }
                np.save(npy_path, temp_data)
                print(f"✓ Temperature data saved to NPY: {len(self.temp_arrays)} frames")
                print(f"  Shape: {temp_data['arrays'].shape}")
            
            actual_duration = time.time() - start_time
            actual_fps = frame_count / actual_duration if actual_duration > 0 else 0
            print(f"\nRecording completed:")
            print(f"  Duration: {actual_duration:.2f}s")
            print(f"  Frames: {frame_count}")
            print(f"  FPS: {actual_fps:.1f}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Record thermal camera data with various formats"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "raw", "both"],
        default="both",
        help="Output format: mp4 (video only), raw (thermal data only), or both (default: both)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frames per second for MP4 video (default: 25.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Output directory for recordings (default: recordings)"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Index of the USB device to use (0 for first device, 1 for second, etc.). If not specified, uses the smallest available device."
    )
    
    args = parser.parse_args()
    
    recorder = EnhancedRecorder(output_dir=args.output_dir, device_index=args.device_index)
    recorder.record(
        duration=args.duration,
        format_type=args.format,
        fps=args.fps
    )
    
    # Cleanup device
    if recorder.device:
        recorder.device.stop()


if __name__ == "__main__":
    main()

