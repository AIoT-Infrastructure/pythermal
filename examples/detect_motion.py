#!/usr/bin/env python3
"""
Example: Motion Detection using Background Subtraction

This example demonstrates how to detect moving objects in thermal images
using background subtraction. It shows both the current frame, background model,
and detected moving objects.
"""

import cv2
import numpy as np
import time
from pythermal import (
    ThermalCapture,
    BackgroundSubtractor,
    detect_moving_objects,
    cluster_objects,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)


def generate_colors(n: int) -> list:
    """Generate distinct colors for visualization."""
    if n == 0:
        return []
    
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    
    return colors


def visualize_motion_detection(
    temp_array: np.ndarray,
    yuyv_frame: np.ndarray,
    background: np.ndarray,
    foreground_mask: np.ndarray,
    objects: list,
    clusters: list,
    min_temp: float,
    max_temp: float
) -> np.ndarray:
    """
    Visualize motion detection with background, foreground mask, and detected objects.
    
    Args:
        temp_array: Current temperature array (96x96)
        yuyv_frame: YUYV frame (240x240)
        background: Background temperature model (96x96)
        foreground_mask: Foreground mask from background subtraction (96x96)
        objects: List of DetectedObject instances
        clusters: List of clusters
        min_temp: Minimum temperature in Celsius
        max_temp: Maximum temperature in Celsius
    
    Returns:
        Visualization image (BGR, 240x240)
    """
    # Convert YUYV to BGR
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Convert temperature arrays to Celsius if needed
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
    background_upscaled = cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    foreground_upscaled = cv2.resize(foreground_mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # Create temperature colormap for current frame
    temp_range = max_temp - min_temp
    if temp_range > 0:
        normalized = ((temp_upscaled - min_temp) / temp_range) * 255.0
        normalized = normalized.clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    temp_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(bgr_frame, 0.4, temp_colored, 0.6, 0)
    
    # Overlay foreground mask (motion areas) in green
    foreground_colored = np.zeros_like(overlay)
    foreground_colored[:, :, 1] = foreground_upscaled  # Green channel
    overlay = cv2.addWeighted(overlay, 0.7, foreground_colored, 0.3, 0)
    
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
            label = f"M{cluster_idx}"
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


def create_background_view(background: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    """Create a visualization of the background model."""
    # Upscale to 240x240
    background_upscaled = cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-255
    temp_range = max_temp - min_temp
    if temp_range > 0:
        normalized = ((background_upscaled - min_temp) / temp_range) * 255.0
        normalized = normalized.clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    # Apply colormap
    background_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Add label
    cv2.putText(
        background_colored,
        "Background Model",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return background_colored


def main():
    """Main function to run the motion detection example"""
    print("Starting thermal motion detection with background subtraction...")
    print("Press 'q' to quit, 'r' to reset background")
    
    # Initialize thermal capture (unified interface for live or recorded)
    print("Initializing thermal capture...")
    capture = None
    try:
        capture = ThermalCapture()  # None/0/empty string defaults to live camera
        
        # Initialize background subtractor
        bg_subtractor = BackgroundSubtractor(
            learning_rate=0.01,  # Slow adaptation for stable background
            min_frames_for_background=15  # Need 15 frames before background is ready
        )
        
        frame_count = 0
        
        while True:
            if not capture.has_new_frame():
                time.sleep(0.01)
                continue
            
            # Get frame data
            metadata = capture.get_metadata()
            if metadata is None:
                continue
            
            temp_array = capture.get_temperature_array()
            yuyv_frame = capture.get_yuyv_frame()
            
            if temp_array is None or yuyv_frame is None:
                capture.mark_frame_read()
                continue
            
            # Detect moving objects
            moving_objects, foreground_mask = detect_moving_objects(
                temp_array=temp_array,
                min_temp=metadata.min_temp,
                max_temp=metadata.max_temp,
                background_subtractor=bg_subtractor,
                temp_threshold=2.0,  # 2°C difference threshold
                min_area=50,
                combine_with_temp_range=True,  # Also filter by temperature range
                temp_min=30.0,  # Human body temperature range
                temp_max=39.0
            )
            
            # Cluster moving objects
            clusters = cluster_objects(moving_objects, max_distance=30.0)
            
            # Get background model
            background = bg_subtractor.get_background()
            if background is None:
                background = np.zeros((TEMP_HEIGHT, TEMP_WIDTH), dtype=np.float32)
            
            # Visualize
            vis_image = visualize_motion_detection(
                temp_array=temp_array,
                yuyv_frame=yuyv_frame,
                background=background,
                foreground_mask=foreground_mask,
                objects=moving_objects,
                clusters=clusters,
                min_temp=metadata.min_temp,
                max_temp=metadata.max_temp
            )
            
            # Add info text
            info_text = [
                f"Frame: {metadata.seq}",
                f"Moving Objects: {len(moving_objects)}, Clusters: {len(clusters)}",
            ]
            if bg_subtractor.is_ready():
                info_text.append("Background: Ready")
            else:
                info_text.append(f"Background: Learning ({bg_subtractor.frame_count}/{bg_subtractor.min_frames_for_background})")
            
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
            cv2.imshow("Motion Detection", vis_image)
            
            # Display background in separate window
            if bg_subtractor.is_ready():
                bg_view = create_background_view(background, metadata.min_temp, metadata.max_temp)
                cv2.imshow("Background Model", bg_view)
            
            # Mark frame as read
            capture.mark_frame_read()
            frame_count += 1
            
            # Check for quit or reset
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting background model...")
                bg_subtractor.reset()
        
        print(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
    except ValueError as e:
        print(f"ERROR: Invalid file format - {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()

