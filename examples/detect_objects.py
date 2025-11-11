#!/usr/bin/env python3
"""
Example: Visualize clustered thermal objects

This example demonstrates how to detect objects in thermal images based on
temperature ranges and visualize them with colored clusters.
"""

import cv2
import numpy as np
from pythermal import (
    ThermalDevice,
    ThermalSharedMemory,
    detect_object_centers,
    cluster_objects,
    DetectedObject,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)


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
    """Main function to run the detection and visualization example"""
    print("Starting thermal object detection visualization...")
    print("Press 'q' to quit")
    
    # Initialize thermal device
    device = ThermalDevice()
    try:
        device.start()
        shm = device.get_shared_memory()
        
        if not shm.initialize():
            print("Failed to initialize shared memory")
            return
        
        frame_count = 0
        
        while True:
            if not shm.has_new_frame():
                continue
            
            # Get frame data
            metadata = shm.get_metadata()
            if metadata is None:
                continue
            
            temp_array = shm.get_temperature_array()
            yuyv_frame = shm.get_yuyv_frame()
            
            if temp_array is None or yuyv_frame is None:
                shm.mark_frame_read()
                continue
            
            # Detect objects (default: 31-39°C for human body detection)
            objects = detect_object_centers(
                temp_array=temp_array,
                min_temp=metadata.min_temp,
                max_temp=metadata.max_temp,
                temp_min=30.0,
                temp_max=39.0,
                min_area=50
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
            
            # Add info text
            info_text = [
                f"Frame: {metadata.seq}",
                f"Objects: {len(objects)}, Clusters: {len(clusters)}",
            ]
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
            cv2.imshow("Thermal Object Detection", vis_image)
            
            # Mark frame as read
            shm.mark_frame_read()
            frame_count += 1
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        print(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        device.stop()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()

