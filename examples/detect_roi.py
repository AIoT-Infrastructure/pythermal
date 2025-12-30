#!/usr/bin/env python3
"""
Example: ROI-based Zone Monitoring

This example demonstrates how to use Region of Interest (ROI) for zone monitoring.
It creates a default center 30x30 ROI and shows how to detect objects only within ROIs.
"""

import cv2
import numpy as np
import time
from pythermal import (
    ThermalCapture,
    detect_object_centers,
    cluster_objects,
    ROIManager,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from pythermal.detections.utils import convert_to_celsius


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


def visualize_roi_detection(
    temp_array: np.ndarray,
    yuyv_frame: np.ndarray,
    roi_manager: ROIManager,
    objects: list,
    clusters: list,
    min_temp: float,
    max_temp: float
) -> np.ndarray:
    """
    Visualize detection with ROI overlays.
    
    Args:
        temp_array: Temperature array (96x96)
        yuyv_frame: YUYV frame (240x240)
        roi_manager: ROIManager instance
        objects: List of DetectedObject instances
        clusters: List of clusters
        min_temp: Minimum temperature in Celsius
        max_temp: Maximum temperature in Celsius
    
    Returns:
        Visualization image (BGR, 240x240)
    """
    # Convert YUYV to BGR
    bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
    
    # Convert temperature array to Celsius if needed
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
    
    # Create temperature colormap
    temp_range = max_temp - min_temp
    if temp_range > 0:
        normalized = ((temp_upscaled - min_temp) / temp_range) * 255.0
        normalized = normalized.clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    
    temp_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(bgr_frame, 0.4, temp_colored, 0.6, 0)
    
    # Draw ROIs
    roi_colors = generate_colors(len(roi_manager.rois))
    for idx, roi in enumerate(roi_manager.rois):
        color = roi_colors[idx] if idx < len(roi_colors) else (255, 255, 255)
        
        # Scale ROI coordinates from 96x96 to 240x240
        x = int((roi.x / TEMP_WIDTH) * WIDTH)
        y = int((roi.y / TEMP_HEIGHT) * HEIGHT)
        w = int((roi.width / TEMP_WIDTH) * WIDTH)
        h = int((roi.height / TEMP_HEIGHT) * HEIGHT)
        
        # Draw ROI rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # Draw ROI label
        label = f"{roi.name}"
        if roi.temp_min is not None or roi.temp_max is not None:
            temp_str = ""
            if roi.temp_min is not None:
                temp_str += f">{roi.temp_min:.0f}C"
            if roi.temp_max is not None:
                if temp_str:
                    temp_str += "-"
                temp_str += f"<{roi.temp_max:.0f}C"
            label += f" ({temp_str})"
        
        cv2.putText(
            overlay,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
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
    """Main function to run the ROI monitoring example"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ROI-based zone monitoring for thermal camera"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index of the USB device to use (0 for first device, 1 for second, etc.). Default: 0"
    )
    args = parser.parse_args()
    
    print("Starting ROI-based zone monitoring...")
    print("Press 'q' to quit")
    
    # Initialize thermal capture (unified interface for live or recorded)
    print("Initializing thermal capture...")
    capture = None
    try:
        capture = ThermalCapture(device_index=args.device_index)  # None/0/empty string defaults to live camera
        
        # Initialize ROI manager
        roi_manager = ROIManager(image_width=TEMP_WIDTH, image_height=TEMP_HEIGHT)
        
        # Add default center 30x30 ROI
        center_roi = roi_manager.add_center_roi(
            size=30,
            name="Center",
            temp_min=30.0,  # Only detect objects above 30°C
            temp_max=39.0   # and below 39°C
        )
        print(f"Added center ROI: {center_roi.x}, {center_roi.y}, {center_roi.width}x{center_roi.height}")
        
        # Optionally add more ROIs
        # roi_manager.add_roi(10, 10, 20, 20, name="TopLeft", temp_min=31.0, temp_max=39.0)
        # roi_manager.add_roi(66, 66, 20, 20, name="BottomRight", temp_min=25.0, temp_max=45.0)
        
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
            
            # Detect all objects
            all_objects = detect_object_centers(
                temp_array=temp_array,
                min_temp=metadata.min_temp,
                max_temp=metadata.max_temp,
                temp_min=30.0,
                temp_max=39.0,
                min_area=50
            )
            
            # Filter objects by ROI
            roi_objects = roi_manager.filter_objects_by_roi(all_objects)
            
            # Further filter by ROI temperature thresholds
            filtered_objects = roi_manager.filter_objects_by_temperature(
                roi_objects,
                convert_to_celsius(temp_array, metadata.min_temp, metadata.max_temp)
            )
            
            # Cluster filtered objects
            clusters = cluster_objects(filtered_objects, max_distance=30.0)
            
            # Get ROI statistics
            temp_celsius = convert_to_celsius(temp_array, metadata.min_temp, metadata.max_temp)
            roi_stats = roi_manager.get_roi_statistics(temp_celsius)
            
            # Visualize
            vis_image = visualize_roi_detection(
                temp_array=temp_array,
                yuyv_frame=yuyv_frame,
                roi_manager=roi_manager,
                objects=filtered_objects,
                clusters=clusters,
                min_temp=metadata.min_temp,
                max_temp=metadata.max_temp
            )
            
            # Add info text
            info_text = [
                f"Frame: {metadata.seq}",
                f"All Objects: {len(all_objects)}, ROI Objects: {len(filtered_objects)}, Clusters: {len(clusters)}",
            ]
            
            # Add ROI statistics
            for roi_name, stats in roi_stats.items():
                info_text.append(f"{roi_name}: {stats['avg']:.0f}C (min:{stats['min']:.0f}, max:{stats['max']:.0f})")
            
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
            cv2.imshow("ROI Zone Monitoring", vis_image)
            
            # Mark frame as read
            capture.mark_frame_read()
            frame_count += 1
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
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

