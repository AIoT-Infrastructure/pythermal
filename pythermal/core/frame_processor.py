#!/usr/bin/env python3
"""
Frame Processor - Process and replay individual thermal frames

Provides functionality to read, write, and process individual thermal camera frames
using the same data structure as .tseq files.
Also supports .tframe format which includes rendered image with overlay.
"""

import struct
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, NamedTuple
from datetime import datetime
from .thermal_shared_memory import FrameMetadata, WIDTH, HEIGHT, TEMP_WIDTH, TEMP_HEIGHT


class ThermalFrame(NamedTuple):
    """Single thermal frame data structure matching .tseq format"""
    timestamp: float
    metadata: FrameMetadata
    yuyv: np.ndarray  # Shape: (HEIGHT, WIDTH, 2), dtype: uint8
    temp_array: np.ndarray  # Shape: (TEMP_HEIGHT, TEMP_WIDTH), dtype: uint16
    rgb: Optional[np.ndarray] = None  # Shape: (HEIGHT, WIDTH, 3), dtype: uint8, optional


class ThermalFrameProcessor:
    """
    Processor for individual thermal camera frames
    
    Provides methods to read, write, and process individual frames
    using the same data structure as .tseq files.
    """
    
    # Frame sizes (matching sequence_reader.py)
    YUYV_SIZE = WIDTH * HEIGHT * 2  # 115200 bytes
    TEMP_SIZE = TEMP_WIDTH * TEMP_HEIGHT * 2  # 18432 bytes
    RGB_SIZE = WIDTH * HEIGHT * 3  # 172800 bytes
    FRAME_HEADER_SIZE = 24  # timestamp (8) + seq (4) + 3 floats (12)
    
    @staticmethod
    def read_frame(file_path: str, frame_index: int = 0, has_color: bool = False) -> Optional[ThermalFrame]:
        """
        Read a single frame from a .tseq file
        
        Args:
            file_path: Path to the .tseq file
            frame_index: Index of frame to read (0-based)
            has_color: Whether the file contains RGB data
        
        Returns:
            ThermalFrame object, or None if frame not found
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate frame size
        frame_size = ThermalFrameProcessor.FRAME_HEADER_SIZE + ThermalFrameProcessor.YUYV_SIZE + ThermalFrameProcessor.TEMP_SIZE
        if has_color:
            frame_size += ThermalFrameProcessor.RGB_SIZE
        
        # Calculate file offset
        header_size = 6  # "TSEQ" + version (1) + color flag (1)
        frame_offset = header_size + frame_index * frame_size
        
        try:
            with open(file_path, "rb") as f:
                # Verify header
                header = f.read(6)
                if len(header) != 6 or header[:4] != b"TSEQ":
                    raise ValueError(f"Invalid file format: {file_path}")
                
                # Seek to frame
                f.seek(frame_offset)
                
                # Read frame header
                frame_header = f.read(ThermalFrameProcessor.FRAME_HEADER_SIZE)
                if len(frame_header) < ThermalFrameProcessor.FRAME_HEADER_SIZE:
                    return None
                
                # Unpack frame header
                timestamp, seq, min_temp, max_temp, avg_temp = struct.unpack("dIfff", frame_header)
                
                # Read YUYV data
                yuyv_bytes = f.read(ThermalFrameProcessor.YUYV_SIZE)
                if len(yuyv_bytes) < ThermalFrameProcessor.YUYV_SIZE:
                    return None
                yuyv = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
                
                # Read temperature array
                temp_bytes = f.read(ThermalFrameProcessor.TEMP_SIZE)
                if len(temp_bytes) < ThermalFrameProcessor.TEMP_SIZE:
                    return None
                temp_array = np.frombuffer(temp_bytes, dtype=np.uint16).reshape((TEMP_HEIGHT, TEMP_WIDTH))
                
                # Read RGB if present
                rgb = None
                if has_color:
                    rgb_bytes = f.read(ThermalFrameProcessor.RGB_SIZE)
                    if len(rgb_bytes) >= ThermalFrameProcessor.RGB_SIZE:
                        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Create metadata
                metadata = FrameMetadata(
                    seq=seq,
                    flag=1,  # Always 1 for recorded frames
                    width=WIDTH,
                    height=HEIGHT,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    avg_temp=avg_temp
                )
                
                return ThermalFrame(
                    timestamp=timestamp,
                    metadata=metadata,
                    yuyv=yuyv,
                    temp_array=temp_array,
                    rgb=rgb
                )
        
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    @staticmethod
    def write_frame(file_path: str, frame: ThermalFrame, has_color: bool = False, 
                    version: int = 1, append: bool = False) -> bool:
        """
        Write a single frame to a .tseq file
        
        Args:
            file_path: Path to the .tseq file
            frame: ThermalFrame object to write
            has_color: Whether to include RGB data
            version: File format version (default: 1)
            append: If True, append to existing file; if False, create new file
        
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        try:
            # Determine file mode
            if append and file_path.exists():
                # Append mode: open existing file
                mode = "ab"
                write_header = False
            else:
                # Create new file
                mode = "wb"
                write_header = True
            
            with open(file_path, mode) as f:
                # Write header if creating new file
                if write_header:
                    header = b"TSEQ" + bytes([version]) + (b"\x01" if has_color else b"\x00")
                    f.write(header)
                
                # Write frame header
                frame_header = struct.pack("dIfff", 
                                         frame.timestamp,
                                         frame.metadata.seq,
                                         frame.metadata.min_temp,
                                         frame.metadata.max_temp,
                                         frame.metadata.avg_temp)
                f.write(frame_header)
                
                # Write YUYV data
                if frame.yuyv.shape != (HEIGHT, WIDTH, 2):
                    raise ValueError(f"Invalid YUYV shape: {frame.yuyv.shape}, expected ({HEIGHT}, {WIDTH}, 2)")
                f.write(frame.yuyv.tobytes())
                
                # Write temperature array
                if frame.temp_array.shape != (TEMP_HEIGHT, TEMP_WIDTH):
                    raise ValueError(f"Invalid temp_array shape: {frame.temp_array.shape}, expected ({TEMP_HEIGHT}, {TEMP_WIDTH})")
                f.write(frame.temp_array.tobytes())
                
                # Write RGB if requested
                if has_color:
                    if frame.rgb is not None:
                        if frame.rgb.shape != (HEIGHT, WIDTH, 3):
                            raise ValueError(f"Invalid RGB shape: {frame.rgb.shape}, expected ({HEIGHT}, {WIDTH}, 3)")
                        f.write(frame.rgb.tobytes())
                    else:
                        # Generate RGB from YUYV if not provided
                        import cv2
                        rgb = cv2.cvtColor(frame.yuyv, cv2.COLOR_YUV2RGB_YUYV)
                        f.write(rgb.tobytes())
            
            return True
        
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False
    
    @staticmethod
    def create_frame_from_capture(capture, timestamp: Optional[float] = None) -> Optional[ThermalFrame]:
        """
        Create a ThermalFrame from a ThermalCapture object
        
        Args:
            capture: ThermalCapture instance (live or recorded)
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            ThermalFrame object, or None if frame data unavailable
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # Get metadata
        metadata = capture.get_metadata()
        if metadata is None:
            return None
        
        # Get frame data
        yuyv = capture.get_yuyv_frame()
        temp_array = capture.get_temperature_array()
        
        if yuyv is None or temp_array is None:
            return None
        
        # Generate RGB from YUYV
        import cv2
        rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
        
        return ThermalFrame(
            timestamp=timestamp,
            metadata=metadata,
            yuyv=yuyv.copy(),
            temp_array=temp_array.copy(),
            rgb=rgb
        )
    
    @staticmethod
    def replay_frame(frame: ThermalFrame, window_name: str = "Frame Replay", 
                     view_mode: str = 'yuyv', wait_key: int = 0) -> int:
        """
        Replay a single frame in an OpenCV window
        
        Args:
            frame: ThermalFrame object to display
            window_name: Name of the OpenCV window
            view_mode: 'yuyv', 'temperature', or 'rgb'
            wait_key: OpenCV waitKey delay (0 = wait for key press)
        
        Returns:
            Key code from waitKey
        """
        import cv2
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 480)
        
        # Render based on view mode
        if view_mode == 'yuyv':
            # Convert YUYV to grayscale BGR
            y_channel = frame.yuyv[:, :, 0]
            image = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        
        elif view_mode == 'temperature':
            # Convert temperature array to colorized view
            temp_float = frame.temp_array.astype(np.float32)
            raw_min = np.min(temp_float)
            raw_max = np.max(temp_float)
            raw_range = raw_max - raw_min
            
            if raw_range > 0:
                normalized = ((temp_float - raw_min) / raw_range) * 255.0
                normalized = normalized.clip(0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(frame.temp_array, dtype=np.uint8)
            
            # Upscale from 96x96 to 240x240
            upscaled = cv2.resize(normalized, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            image = cv2.applyColorMap(upscaled, cv2.COLORMAP_HOT)
        
        elif view_mode == 'rgb' and frame.rgb is not None:
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        
        else:
            # Fallback to YUYV grayscale
            y_channel = frame.yuyv[:, :, 0]
            image = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        
        # Add text overlay with metadata
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        info_text = [
            f"Seq: {frame.metadata.seq}",
            f"Min: {frame.metadata.min_temp:.1f}C",
            f"Max: {frame.metadata.max_temp:.1f}C",
            f"Avg: {frame.metadata.avg_temp:.1f}C",
            f"Time: {datetime.fromtimestamp(frame.timestamp).strftime('%H:%M:%S')}"
        ]
        
        y_offset = 20
        for text in info_text:
            cv2.putText(image, text, (5, y_offset), font, font_scale, color, thickness)
            y_offset += 20
        
        # Display image
        cv2.imshow(window_name, image)
        return cv2.waitKey(wait_key) & 0xFF
    
    @staticmethod
    def get_frame_count(file_path: str, has_color: bool = False) -> int:
        """
        Get the number of frames in a .tseq file
        
        Args:
            file_path: Path to the .tseq file
            has_color: Whether the file contains RGB data
        
        Returns:
            Number of frames in the file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return 0
        
        # Calculate frame size
        frame_size = ThermalFrameProcessor.FRAME_HEADER_SIZE + ThermalFrameProcessor.YUYV_SIZE + ThermalFrameProcessor.TEMP_SIZE
        if has_color:
            frame_size += ThermalFrameProcessor.RGB_SIZE
        
        # Calculate total frames
        header_size = 6
        file_size = file_path.stat().st_size
        total_frames = (file_size - header_size) // frame_size
        
        return max(0, total_frames)
    
    @staticmethod
    def write_tframe(file_path: str, rendered_image: np.ndarray, frame: ThermalFrame, 
                     view_mode: str = 'yuyv', version: int = 1) -> bool:
        """
        Write a thermal frame as .tframe file (includes rendered image + raw data)
        
        Args:
            file_path: Path to save .tframe file
            rendered_image: BGR image with overlay (can be any size)
            frame: ThermalFrame object with raw data
            view_mode: View mode used ('yuyv', 'temperature', 'temperature_celsius')
            version: File format version (default: 1)
        
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, "wb") as f:
                # Write header: "TFRM" + version (1 byte)
                f.write(b"TFRM" + bytes([version]))
                
                # Write view mode (1 byte length + string)
                view_mode_bytes = view_mode.encode('utf-8')
                if len(view_mode_bytes) > 255:
                    raise ValueError(f"View mode string too long: {len(view_mode_bytes)}")
                f.write(bytes([len(view_mode_bytes)]) + view_mode_bytes)
                
                # Write timestamp
                f.write(struct.pack("d", frame.timestamp))
                
                # Write metadata
                f.write(struct.pack("Ifff", 
                                   frame.metadata.seq,
                                   frame.metadata.min_temp,
                                   frame.metadata.max_temp,
                                   frame.metadata.avg_temp))
                
                # Write rendered image size and data
                # Encode image as PNG for compression
                success, img_encoded = cv2.imencode('.png', rendered_image)
                if not success:
                    raise ValueError("Failed to encode rendered image")
                
                img_size = len(img_encoded)
                f.write(struct.pack("I", img_size))
                f.write(img_encoded.tobytes())
                
                # Write YUYV data
                if frame.yuyv.shape != (HEIGHT, WIDTH, 2):
                    raise ValueError(f"Invalid YUYV shape: {frame.yuyv.shape}, expected ({HEIGHT}, {WIDTH}, 2)")
                f.write(frame.yuyv.tobytes())
                
                # Write temperature array
                if frame.temp_array.shape != (TEMP_HEIGHT, TEMP_WIDTH):
                    raise ValueError(f"Invalid temp_array shape: {frame.temp_array.shape}, expected ({TEMP_HEIGHT}, {TEMP_WIDTH})")
                f.write(frame.temp_array.tobytes())
                
                # Write RGB if available
                if frame.rgb is not None:
                    f.write(b"\x01")  # Flag: RGB present
                    if frame.rgb.shape != (HEIGHT, WIDTH, 3):
                        raise ValueError(f"Invalid RGB shape: {frame.rgb.shape}, expected ({HEIGHT}, {WIDTH}, 3)")
                    f.write(frame.rgb.tobytes())
                else:
                    f.write(b"\x00")  # Flag: RGB not present
            
            return True
        
        except Exception as e:
            print(f"Error writing .tframe file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def read_tframe(file_path: str) -> Optional[dict]:
        """
        Read a .tframe file
        
        Args:
            file_path: Path to .tframe file
        
        Returns:
            Dictionary with keys:
            - 'rendered_image': BGR image with overlay
            - 'frame': ThermalFrame object
            - 'view_mode': View mode string
            - 'timestamp': Timestamp
            - 'metadata': FrameMetadata
            Or None if error
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "rb") as f:
                # Read header
                header = f.read(5)
                if len(header) != 5 or header[:4] != b"TFRM":
                    raise ValueError(f"Invalid .tframe file format: {file_path}")
                
                version = header[4]
                if version != 1:
                    raise ValueError(f"Unsupported .tframe version: {version}")
                
                # Read view mode
                view_mode_len = ord(f.read(1))
                view_mode = f.read(view_mode_len).decode('utf-8')
                
                # Read timestamp
                timestamp_bytes = f.read(8)
                if len(timestamp_bytes) < 8:
                    return None
                timestamp = struct.unpack("d", timestamp_bytes)[0]
                
                # Read metadata
                metadata_bytes = f.read(16)  # seq (4) + 3 floats (12)
                if len(metadata_bytes) < 16:
                    return None
                seq, min_temp, max_temp, avg_temp = struct.unpack("Ifff", metadata_bytes)
                
                metadata = FrameMetadata(
                    seq=seq,
                    flag=1,
                    width=WIDTH,
                    height=HEIGHT,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    avg_temp=avg_temp
                )
                
                # Read rendered image size and data
                img_size_bytes = f.read(4)
                if len(img_size_bytes) < 4:
                    return None
                img_size = struct.unpack("I", img_size_bytes)[0]
                
                img_encoded = f.read(img_size)
                if len(img_encoded) < img_size:
                    return None
                
                # Decode PNG image
                img_array = np.frombuffer(img_encoded, dtype=np.uint8)
                rendered_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if rendered_image is None:
                    raise ValueError("Failed to decode rendered image")
                
                # Read YUYV data
                yuyv_bytes = f.read(ThermalFrameProcessor.YUYV_SIZE)
                if len(yuyv_bytes) < ThermalFrameProcessor.YUYV_SIZE:
                    return None
                yuyv = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
                
                # Read temperature array
                temp_bytes = f.read(ThermalFrameProcessor.TEMP_SIZE)
                if len(temp_bytes) < ThermalFrameProcessor.TEMP_SIZE:
                    return None
                temp_array = np.frombuffer(temp_bytes, dtype=np.uint16).reshape((TEMP_HEIGHT, TEMP_WIDTH))
                
                # Read RGB flag and data if present
                rgb_flag = f.read(1)
                rgb = None
                if len(rgb_flag) > 0 and rgb_flag[0] == 1:
                    rgb_bytes = f.read(ThermalFrameProcessor.RGB_SIZE)
                    if len(rgb_bytes) >= ThermalFrameProcessor.RGB_SIZE:
                        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                frame = ThermalFrame(
                    timestamp=timestamp,
                    metadata=metadata,
                    yuyv=yuyv,
                    temp_array=temp_array,
                    rgb=rgb
                )
                
                return {
                    'rendered_image': rendered_image,
                    'frame': frame,
                    'view_mode': view_mode,
                    'timestamp': timestamp,
                    'metadata': metadata
                }
        
        except Exception as e:
            print(f"Error reading .tframe file: {e}")
            import traceback
            traceback.print_exc()
            return None

