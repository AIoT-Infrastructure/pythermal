#!/usr/bin/env python3
"""
Example: Live YOLO v11 Pose Detection on Thermal Images

This example demonstrates real-time pose/keypoint detection using YOLO v11
on thermal camera feed. It detects human poses and visualizes keypoints
and skeleton connections.
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
    from pythermal.detections.yolo import YOLOPoseDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: YOLO pose detection requires ultralytics package.")
    print("Install it with: pip install ultralytics")
    exit(1)


def main():
    """Main function to run live pose detection"""
    print("Starting YOLO v11 Pose Detection on Thermal Camera...")
    print("Press 'q' to quit")
    print("Press 'b' to toggle bounding boxes")
    print("Press 'k' to toggle keypoints")
    print("Press 's' to toggle skeleton")
    print("Press 'l' to toggle labels")
    
    # Initialize thermal device
    device = ThermalDevice()
    try:
        device.start()
        shm = device.get_shared_memory()
        
        if not shm.initialize():
            print("Failed to initialize shared memory")
            return
        
        # Initialize YOLO pose detector
        # Use default nano model for edge devices
        # For custom thermal model, use: model_path="custom_thermal_pose.pt"
        print("Loading YOLO v11 pose model (this may take a moment on first run)...")
        detector = YOLOPoseDetector(
            model_path="./pythermal/detections/yolo/models/yolov11-pose-best.pt",
            #model_size="nano",  # Options: "nano", "small", "medium", "large", "xlarge"
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
        print("Model loaded successfully!")
        
        # Visualization settings
        show_bbox = True
        show_keypoints = True
        show_skeleton = True
        show_labels = False
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0
        
        window_name = "YOLO Pose Detection - Thermal Camera"
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
            
            # Run pose detection
            detections = detector.detect(bgr_frame, verbose=False)
            
            # Visualize detections
            vis_image = detector.visualize(
                bgr_frame,
                detections,
                show_bbox=show_bbox,
                show_keypoints=show_keypoints,
                show_skeleton=show_skeleton,
                show_labels=show_labels,
                keypoint_radius=3,
                skeleton_thickness=2,
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
                f"Poses detected: {len(detections)}",
            ]
            
            # Add pose details
            for i, det in enumerate(detections):
                info_lines.append(f"Pose {i+1}: conf={det['confidence']:.2f}")
            
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
                "q=quit, b=bbox, k=keypoints",
                "s=skeleton, l=labels",
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
            elif key == ord('b'):
                show_bbox = not show_bbox
                print(f"Bounding boxes: {'ON' if show_bbox else 'OFF'}")
            elif key == ord('k'):
                show_keypoints = not show_keypoints
                print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
            elif key == ord('s'):
                show_skeleton = not show_skeleton
                print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels: {'ON' if show_labels else 'OFF'}")
        
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

