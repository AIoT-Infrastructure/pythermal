#!/usr/bin/env python3
"""
Example: Live YOLO v11 Pose Detection on Thermal Images

This example demonstrates real-time pose/keypoint detection using YOLO v11
on thermal camera feed. It detects human poses and visualizes keypoints
and skeleton connections. Supports both live camera and recorded sequences.
"""

import cv2
import numpy as np
import argparse
import time
from pythermal import (
    ThermalCapture,
    WIDTH,
    HEIGHT,
    detect_humans_adaptive,
)
from pythermal.utils import estimate_environment_temperature_v1

try:
    from pythermal.detections.yolo import YOLOPoseDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: YOLO pose detection requires ultralytics package.")
    print("Install it with: pip install ultralytics")
    exit(1)


def filter_poses_by_temperature(
    pose_detections,
    bgr_frame,
    temp_array,
    min_temp,
    max_temp,
    environment_temp=None,
    min_temp_above_env=2.0,
    max_temp_limit=42.0
):
    """
    Filter pose detections based on temperature to keep only actual humans.
    
    Args:
        pose_detections: List of pose detections from YOLO
        bgr_frame: BGR frame for coordinate mapping
        temp_array: Temperature array (96x96)
        min_temp: Minimum temperature from metadata
        max_temp: Maximum temperature from metadata
        environment_temp: Environment temperature (None to auto-estimate)
        min_temp_above_env: Minimum temp above environment
        max_temp_limit: Maximum temperature limit
    
    Returns:
        Filtered list of pose detections
    """
    if not pose_detections or temp_array is None:
        return pose_detections
    
    # Estimate environment temperature if not provided
    if environment_temp is None:
        environment_temp = estimate_environment_temperature_v1(
            temp_array, min_temp, max_temp
        )
        if environment_temp is None:
            return pose_detections
    
    # Get temperature-based human detections
    temp_objects = detect_humans_adaptive(
        temp_array=temp_array,
        min_temp=min_temp,
        max_temp=max_temp,
        environment_temp=environment_temp,
        min_temp_above_env=min_temp_above_env,
        max_temp_limit=max_temp_limit,
        alpha_min=0.4,
        alpha_max=0.7
    )
    
    if not temp_objects:
        return []
    
    # Scale factors for coordinate conversion
    # YOLO detections are in BGR frame coordinates (240x240)
    # Temperature array is 96x96
    temp_to_bgr_scale_x = WIDTH / 96.0
    temp_to_bgr_scale_y = HEIGHT / 96.0
    
    filtered_poses = []
    for pose in pose_detections:
        # Get face keypoint (nose) from pose detection
        # Prefer nose as it's the most central face keypoint
        keypoints_dict = pose.get('keypoints_dict', {})
        face_x, face_y = None, None
        
        # Try to get nose keypoint first
        if 'nose' in keypoints_dict:
            nose_data = keypoints_dict['nose']
            if len(nose_data) >= 2 and nose_data[2] > 0:  # Check confidence > 0
                face_x, face_y = nose_data[0], nose_data[1]
        
        # Fallback to other face keypoints if nose not available
        if face_x is None or face_y is None:
            for face_keypoint in ['left_eye', 'right_eye', 'left_ear', 'right_ear']:
                if face_keypoint in keypoints_dict:
                    kp_data = keypoints_dict[face_keypoint]
                    if len(kp_data) >= 2 and kp_data[2] > 0:  # Check confidence > 0
                        face_x, face_y = kp_data[0], kp_data[1]
                        break
        
        # If no face keypoint available, fall back to center
        if face_x is None or face_y is None:
            center = pose.get('center', None)
            if center is None:
                continue
            face_x, face_y = center
        
        # Convert face keypoint to temperature array coordinates
        temp_x = int(face_x / temp_to_bgr_scale_x)
        temp_y = int(face_y / temp_to_bgr_scale_y)
        
        # Check if face keypoint is within any temperature-detected human region
        is_human = False
        for temp_obj in temp_objects:
            # Check if face keypoint is within temperature object bounds
            obj_x_min = temp_obj.center_x - temp_obj.width / 2
            obj_x_max = temp_obj.center_x + temp_obj.width / 2
            obj_y_min = temp_obj.center_y - temp_obj.height / 2
            obj_y_max = temp_obj.center_y + temp_obj.height / 2
            
            if (obj_x_min <= temp_x <= obj_x_max and 
                obj_y_min <= temp_y <= obj_y_max):
                is_human = True
                break
        
        if is_human:
            filtered_poses.append(pose)
    
    return filtered_poses


def main():
    """Main function to run pose detection on live or recorded data"""
    parser = argparse.ArgumentParser(
        description="YOLO v11 Pose Detection on Thermal Images (live or recorded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live detection (default)
  python yolo_pose_detection.py
  
  # Recorded detection
  python yolo_pose_detection.py recordings/thermal_20240101.tseq
  
  # Recorded with custom FPS
  python yolo_pose_detection.py file.tseq --fps 30
  
  # Recorded with temperature filtering
  python yolo_pose_detection.py file.tseq --temp-filter --env-temp 22.0
  
  # Recorded with temperature filtering and custom FPS
  python yolo_pose_detection.py file.tseq --temp-filter --fps 30
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
        "--temp-filter",
        action="store_true",
        help="Enable temperature-based filtering to keep only actual humans (default: False)"
    )
    parser.add_argument(
        "--env-temp",
        type=float,
        default=None,
        help="Environment temperature in Celsius (for temperature filtering). If not provided, will be estimated."
    )
    parser.add_argument(
        "--min-temp-above-env",
        type=float,
        default=2.0,
        help="Minimum temperature above environment for filtering (default: 2.0°C)"
    )
    parser.add_argument(
        "--max-temp-limit",
        type=float,
        default=42.0,
        help="Maximum temperature limit for filtering (default: 42.0°C)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS for recorded data (default: use original timestamps, live: no limit)"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index of the USB device to use (0 for first device, 1 for second, etc.). Default: 0"
    )
    
    args = parser.parse_args()
    
    print("Starting YOLO v11 Pose Detection on Thermal Camera...")
    if args.temp_filter:
        print("Temperature filtering: ENABLED")
    print("Press 'q' to quit")
    print("Press 'b' to toggle bounding boxes")
    print("Press 'k' to toggle keypoints")
    print("Press 's' to toggle skeleton")
    print("Press 'l' to toggle labels")
    if args.source:
        print(f"Source: {args.source}")
    
    # Initialize thermal capture (unified interface for live or recorded)
    print("Initializing thermal capture...")
    capture = None
    try:
        capture = ThermalCapture(args.source, device_index=args.device_index)
        is_recorded = capture.is_recorded
        
        if is_recorded:
            print(f"Starting pose detection on pre-recorded sequence...")
            print(f"File: {args.source}")
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                print(f"Total frames: {int(total_frames)}")
            print("Press 'q' to quit, SPACE to pause/resume")
        else:
            print("Starting pose detection visualization (live)...")
        
        # Initialize YOLO pose detector
        # Use default nano model for edge devices
        # For custom thermal model, use: model_path="custom_thermal_pose.pt"
        print("Loading YOLO v11 pose model (this may take a moment on first run)...")
        detector = YOLOPoseDetector(
            model_size="nano",  # Options: "nano", "small", "medium", "large", "xlarge"
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
        print("Model loaded successfully!")
        
        # Visualization settings
        show_bbox = True
        show_keypoints = True
        show_skeleton = True
        show_labels = False
        
        frame_count = 0  # For FPS calculation (resets every second)
        total_frame_count = 0  # Total frames processed
        fps_start_time = time.time()
        fps = 0.0
        paused = False
        last_display_time = time.time()
        
        window_name = "YOLO Pose Detection - Thermal Camera" + (" (Replay)" if is_recorded else "")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 640)
        
        while True:
            if not paused:
                if not capture.has_new_frame():
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {total_frame_count} frames")
                        break
                    time.sleep(0.01)
                    continue
                
                # Get frame data
                metadata = capture.get_metadata()
                if metadata is None:
                    if is_recorded:
                        break
                    continue
                
                # Read YUYV frame and convert to BGR
                yuyv_frame = capture.get_yuyv_frame()
                temp_array = capture.get_temperature_array()
                if yuyv_frame is None:
                    if is_recorded:
                        break
                    capture.mark_frame_read()
                    continue
                
                # Convert YUYV to BGR for YOLO
                bgr_frame = cv2.cvtColor(yuyv_frame, cv2.COLOR_YUV2BGR_YUYV)
                
                # Run pose detection
                detections = detector.detect(bgr_frame, verbose=False)
                
                # Apply temperature filtering if enabled
                if args.temp_filter and temp_array is not None:
                    detections = filter_poses_by_temperature(
                        detections,
                        bgr_frame,
                        temp_array,
                        metadata.min_temp,
                        metadata.max_temp,
                        environment_temp=args.env_temp,
                        min_temp_above_env=args.min_temp_above_env,
                        max_temp_limit=args.max_temp_limit
                    )
                
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
                if is_recorded:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    info_lines = [
                        f"Frame: {metadata.seq} / {total_frames}",
                        f"FPS: {fps:.1f}",
                        f"Poses detected: {len(detections)}",
                    ]
                else:
                    info_lines = [
                        f"Frame: {metadata.seq}",
                        f"FPS: {fps:.1f}",
                        f"Poses detected: {len(detections)}",
                    ]
                
                if args.temp_filter:
                    info_lines.append("Temp filter: ON")
                
                # Add pose details
                for i, det in enumerate(detections[:5]):  # Limit to first 5 to avoid clutter
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
                capture.mark_frame_read()
                total_frame_count += 1
                
                # Handle playback timing for recorded data
                if is_recorded and args.fps:
                    elapsed = time.time() - last_display_time
                    target_delay = 1.0 / args.fps
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)
                    last_display_time = time.time()
                elif is_recorded:
                    last_display_time = time.time()
            
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
            elif key == ord(' ') and is_recorded:
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        print(f"Processed {total_frame_count} frames")
    
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

