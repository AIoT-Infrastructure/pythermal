#!/usr/bin/env python3
"""
Example: Live YOLO v11 Object Detection on Thermal Images

This example demonstrates real-time object detection using YOLO v11
on thermal camera feed. It detects objects and visualizes bounding boxes
with class labels and confidence scores.
"""

import cv2
import numpy as np
import time
from pythermal import (
    ThermalDevice,
    ThermalSharedMemory,
    WIDTH,
    HEIGHT,
)

try:
    from pythermal.detections.yolo import YOLOObjectDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: YOLO object detection requires ultralytics package.")
    print("Install it with: pip install ultralytics")
    exit(1)


def main():
    """Main function to run live object detection"""
    print("Starting YOLO v11 Object Detection on Thermal Camera...")
    print("Press 'q' to quit")
    print("Press 'l' to toggle labels")
    print("Press 'c' to toggle confidence scores")
    print("Press '+' to increase confidence threshold")
    print("Press '-' to decrease confidence threshold")
    
    # Initialize thermal device
    device = ThermalDevice()
    try:
        device.start()
        shm = device.get_shared_memory()
        
        if not shm.initialize():
            print("Failed to initialize shared memory")
            return
        
        # Initialize YOLO object detector
        # Use default nano model for edge devices
        # For custom thermal model, use: model_path="custom_thermal_object.pt"
        print("Loading YOLO v11 object detection model (this may take a moment on first run)...")
        conf_threshold = 0.25
        detector = YOLOObjectDetector(
            #model_size="nano",  # Options: "nano", "small", "medium", "large", "xlarge"
            model_path="./pythermal/detections/yolo/models/yolov8-object-best.pt",
            conf_threshold=conf_threshold,
            iou_threshold=0.45,
        )
        print("Model loaded successfully!")
        print(f"Available classes: {len(detector.get_class_names())}")
        
        # Visualization settings
        show_labels = True
        show_conf = True
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0
        
        window_name = "YOLO Object Detection - Thermal Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 640)
        
        while True:
            if not shm.has_new_frame():
                time.sleep(0.01)
                continue
            
            # Get frame data
            metadata = shm.get_metadata()
            if metadata is None:
                continue
            
            # Read YUYV frame and convert to BGR
            yuyv_frame = shm.get_yuyv_frame()
            if yuyv_frame is None:
                shm.mark_frame_read()
                continue
            
            # Convert YUYV to BGR for YOLO
            bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
            
            # Update confidence threshold if changed
            if detector.conf_threshold != conf_threshold:
                detector.conf_threshold = conf_threshold
            
            # Run object detection
            detections = detector.detect(bgr_frame, verbose=False)
            
            # Visualize detections
            vis_image = detector.visualize(
                bgr_frame,
                detections,
                show_labels=show_labels,
                show_conf=show_conf,
            )
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            # Add info overlay
            info_lines = [
                f"Frame: {metadata.seq}",
                f"FPS: {fps:.1f}",
                f"Conf threshold: {conf_threshold:.2f}",
                f"Objects detected: {len(detections)}",
            ]
            
            # Count objects by class
            class_counts = {}
            for det in detections:
                class_name = det["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Add class counts
            for class_name, count in sorted(class_counts.items()):
                info_lines.append(f"  {class_name}: {count}")
            
            y_offset = 15
            for line in info_lines:
                cv2.putText(
                    vis_image,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                y_offset += 15
            
            # Add control hints
            control_text = [
                "Controls:",
                "q=quit, l=labels, c=confidence",
                "+/-=adjust threshold",
            ]
            y_offset = HEIGHT - 45
            for line in control_text:
                cv2.putText(
                    vis_image,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                )
                y_offset += 12
            
            # Display image
            cv2.imshow(window_name, vis_image)
            
            # Mark frame as read
            shm.mark_frame_read()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('c'):
                show_conf = not show_conf
                print(f"Confidence scores: {'ON' if show_conf else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(1.0, conf_threshold + 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.0, conf_threshold - 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
        
        print(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        device.stop()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()

