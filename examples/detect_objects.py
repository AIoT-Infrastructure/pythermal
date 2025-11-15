#!/usr/bin/env python3
"""
Example: Visualize clustered thermal objects (Live or Recorded)

This example demonstrates how to detect objects in thermal images based on
temperature ranges and visualize them with colored clusters.
Supports both live camera feed and pre-recorded sequences using ThermalCapture.
"""

import cv2
import numpy as np
import argparse
import time
from pythermal import (
    ThermalCapture,
    detect_object_centers,
    detect_humans_adaptive,
    cluster_objects,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from pythermal.utils import estimate_environment_temperature_v1


def generate_colors(n: int) -> list:
    """
    Generate distinct colors for visualization.
    
    Args:
        n: Number of colors needed
    
    Returns:
        List of BGR color tuples
    """
    if n == 0:
        return []
    
    # Use HSV color space for better color distribution
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    
    return colors


def visualize_clustered_objects(
    temp_array: np.ndarray,
    yuyv_frame: np.ndarray,
    objects: list,
    clusters: list,
    min_temp: float,
    max_temp: float
) -> np.ndarray:
    """
    Visualize detected objects with colored clusters on the thermal image.
    
    Args:
        temp_array: Temperature array (96x96)
        yuyv_frame: YUYV frame (240x240)
        objects: List of DetectedObject instances
        clusters: List of clusters (each cluster is a list of DetectedObject)
        min_temp: Minimum temperature in Celsius
        max_temp: Maximum temperature in Celsius
    
    Returns:
        Visualization image (BGR, 240x240)
    """
    # Convert YUYV to BGR for display
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Upscale temperature array to match frame size for visualization
    temp_celsius = temp_array
    if temp_array.dtype == np.uint16:
        raw_min = np.min(temp_array)
        raw_max = np.max(temp_array)
        raw_range = raw_max - raw_min
        if raw_range > 0:
            normalized = (temp_array.astype(np.float32) - raw_min) / raw_range
            temp_celsius = min_temp + normalized * (max_temp - min_temp)
        else:
            temp_celsius = np.full_like(temp_array, (min_temp + max_temp) / 2.0, dtype=np.float32)
    
    # Upscale to 240x240
    temp_upscaled = cv2.resize(temp_celsius, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Create overlay with temperature colormap
    temp_range = max_temp - min_temp
    if temp_range > 0:
        normalized = ((temp_upscaled - min_temp) / temp_range) * 255.0
        normalized = normalized.clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    # Apply colormap
    temp_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(bgr_frame, 0.5, temp_colored, 0.5, 0)
    
    # Generate colors for clusters
    cluster_colors = generate_colors(len(clusters))
    
    # Draw clusters with different colors
    for cluster_idx, cluster in enumerate(clusters):
        color = cluster_colors[cluster_idx] if cluster_idx < len(cluster_colors) else (255, 255, 255)
        
        for obj in cluster:
            # Scale coordinates from 96x96 to 240x240
            center_x = int((obj.center_x / TEMP_WIDTH) * WIDTH)
            center_y = int((obj.center_y / TEMP_HEIGHT) * HEIGHT)
            width = int((obj.width / TEMP_WIDTH) * WIDTH)
            height = int((obj.height / TEMP_HEIGHT) * HEIGHT)
            
            # Draw bounding box
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
            
            # Draw center point
            cv2.circle(overlay, (center_x, center_y), 5, color, -1)
            
            # Draw cluster ID
            label = f"C{cluster_idx}"
            cv2.putText(
                overlay,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw temperature info
            temp_label = f"{obj.avg_temperature:.0f}C"
            cv2.putText(
                overlay,
                temp_label,
                (x, y + height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
    
    return overlay


def main():
    """Main function to run detection on live or recorded data"""
    parser = argparse.ArgumentParser(
        description="Detect objects in thermal images (live or recorded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live detection (default)
  python detect_objects.py
  python detect_objects.py 0
  
  # Recorded detection
  python detect_objects.py recordings/thermal_20240101.tseq
  
  # Recorded with custom parameters
  python detect_objects.py file.tseq --temp-min 25.0 --temp-max 40.0 --fps 30
  
  # Adaptive human detection (automatically adjusts based on room temperature)
  python detect_objects.py --adaptive
  
  # Adaptive detection with known room temperature
  python detect_objects.py --adaptive --env-temp 22.0
  
  # Face-only detection (warmer body parts, α ≈ 0.5–0.7)
  python detect_objects.py --adaptive --face-only
  
  # Face-only with known room temperature
  python detect_objects.py --adaptive --face-only --env-temp 22.0
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
        "--temp-min",
        type=float,
        default=30.0,
        help="Minimum temperature for detection in Celsius (default: 30.0)"
    )
    parser.add_argument(
        "--temp-max",
        type=float,
        default=39.0,
        help="Maximum temperature for detection in Celsius (default: 39.0)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum area for detection (default: 50)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS for recorded data (default: use original timestamps, live: no limit)"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive human detection based on environment temperature (default: False)"
    )
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="Detect faces only (warmer body parts, α ≈ 0.5–0.7). Requires --adaptive flag."
    )
    parser.add_argument(
        "--env-temp",
        type=float,
        default=None,
        help="Environment temperature in Celsius (for adaptive detection). If not provided, will be estimated from frame."
    )
    parser.add_argument(
        "--min-temp-above-env",
        type=float,
        default=2.0,
        help="Minimum temperature above environment for adaptive detection (default: 2.0°C)"
    )
    parser.add_argument(
        "--max-temp-limit",
        type=float,
        default=42.0,
        help="Maximum temperature limit to avoid detecting hot objects (default: 42.0°C)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.face_only and not args.adaptive:
        print("ERROR: --face-only requires --adaptive flag")
        return
    
    # Initialize thermal capture (unified interface)
    print("Initializing thermal capture...")
    try:
        capture = ThermalCapture(args.source)
        is_recorded = capture.is_recorded
        
        if is_recorded:
            print("Starting object detection on pre-recorded sequence...")
            print(f"File: {args.source}")
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                print(f"Total frames: {int(total_frames)}")
            print("Press 'q' to quit, SPACE to pause/resume")
        else:
            print("Starting thermal object detection visualization (live)...")
            print("Press 'q' to quit")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    except ValueError as e:
        print(f"ERROR: Invalid file format - {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize thermal capture: {e}")
        return
    
    paused = False
    frame_count = 0
    last_display_time = time.time()
    window_name = "Thermal Object Detection" + (" (Replay)" if is_recorded else "")
    
    try:
        while True:
            if not paused:
                # Check for new frame
                if not capture.has_new_frame():
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {frame_count} frames")
                        break
                    # Live data - wait a bit
                    time.sleep(0.01)
                    continue
                
                # Get frame data using unified interface
                # For recorded files, get_metadata() will automatically read the frame if needed
                metadata = capture.get_metadata()
                temp_array = capture.get_temperature_array()
                yuyv_frame = capture.get_yuyv_frame()
                
                if metadata is None or temp_array is None or yuyv_frame is None:
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {frame_count} frames")
                        break
                    continue
                
                # Detect objects using either standard or adaptive method
                if args.adaptive:
                    # Use adaptive human detection based on environment temperature
                    if args.face_only:
                        # Face-only detection: use narrower alpha range (0.5-0.7) for warmer body parts
                        objects = detect_humans_adaptive(
                            temp_array=temp_array,
                            min_temp=metadata.min_temp,
                            max_temp=metadata.max_temp,
                            environment_temp=args.env_temp,
                            min_area=args.min_area,
                            min_temp_above_env=args.min_temp_above_env,
                            max_temp_limit=args.max_temp_limit,
                            alpha_min=0.5,  # Face/torso range
                            alpha_max=0.7   # Face/torso range
                        )
                    else:
                        # Full body detection: includes cooler parts (0.4-0.7)
                        objects = detect_humans_adaptive(
                            temp_array=temp_array,
                            min_temp=metadata.min_temp,
                            max_temp=metadata.max_temp,
                            environment_temp=args.env_temp,
                            min_area=args.min_area,
                            min_temp_above_env=args.min_temp_above_env,
                            max_temp_limit=args.max_temp_limit
                        )
                else:
                    # Use standard temperature-based detection
                    objects = detect_object_centers(
                        temp_array=temp_array,
                        min_temp=metadata.min_temp,
                        max_temp=metadata.max_temp,
                        temp_min=args.temp_min,
                        temp_max=args.temp_max,
                        min_area=args.min_area
                    )
                
                # Cluster objects
                clusters = cluster_objects(objects, max_distance=30.0)
                
                # Visualize
                vis_image = visualize_clustered_objects(
                    temp_array=temp_array,
                    yuyv_frame=yuyv_frame,
                    objects=objects,
                    clusters=clusters,
                    min_temp=metadata.min_temp,
                    max_temp=metadata.max_temp
                )
                
                # Estimate environment temperature for display (if using adaptive)
                env_temp_display = None
                if args.adaptive:
                    if args.env_temp is not None:
                        env_temp_display = args.env_temp
                    else:
                        env_temp_display = estimate_environment_temperature_v1(
                            temp_array, metadata.min_temp, metadata.max_temp
                        )
                
                # Add info text
                if is_recorded:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    info_text = [
                        f"Frame: {metadata.seq} / {total_frames}",
                        f"Objects: {len(objects)}, Clusters: {len(clusters)}",
                    ]
                else:
                    info_text = [
                        f"Frame: {metadata.seq}",
                        f"Objects: {len(objects)}, Clusters: {len(clusters)}",
                    ]
                
                # Add detection method info
                if args.adaptive:
                    method_text = "Adaptive Detection"
                    if args.face_only:
                        method_text += " [Face Only]"
                    if env_temp_display is not None:
                        method_text += f" (Room: {env_temp_display:.1f}C)"
                    info_text.append(method_text)
                else:
                    info_text.append(f"Range: {args.temp_min:.0f}C - {args.temp_max:.0f}C")
                
                # Add detected object temperature range if objects found
                if objects:
                    obj_temps = [obj.avg_temperature for obj in objects]
                    info_text.append(f"Detected: {min(obj_temps):.0f}C - {max(obj_temps):.0f}C")
                
                y_offset = 15
                for text in info_text:
                    cv2.putText(
                        vis_image,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
                    y_offset += 18
                
                # Display
                cv2.imshow(window_name, vis_image)
                
                # Mark frame as read
                capture.mark_frame_read()
                frame_count += 1
                
                # Handle playback timing for recorded data
                if is_recorded and args.fps:
                    elapsed = time.time() - last_display_time
                    target_delay = 1.0 / args.fps
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)
                    last_display_time = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and is_recorded:
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        print(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
