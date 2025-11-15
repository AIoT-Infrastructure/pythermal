#!/usr/bin/env python3
"""
Example: Live YOLO Human Detection on Thermal Images

This example demonstrates real-time human detection using YOLOv8 thermal model
from Hugging Face (pitangent-ds/YOLOv8-human-detection-thermal) on thermal camera feed.
It detects humans and visualizes bounding boxes with class labels and confidence scores.
Supports both live camera and recorded sequences.
"""

import cv2
import numpy as np
import argparse
import time
from pythermal import (
    ThermalCapture,
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

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install it with: pip install huggingface_hub")


def main():
    """Main function to run human detection on live or recorded data"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Human Detection on Thermal Images (live or recorded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live detection (default)
  python yolo_object_detection.py
  
  # Recorded detection
  python yolo_object_detection.py recordings/thermal_20240101.tseq
  
  # Recorded with custom FPS
  python yolo_object_detection.py file.tseq --fps 30
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
        "--fps",
        type=float,
        default=None,
        help="Playback FPS for recorded data (default: use original timestamps, live: no limit)"
    )
    
    args = parser.parse_args()
    
    print("Starting YOLOv8 Human Detection on Thermal Camera...")
    print("Press 'q' to quit")
    print("Press 'l' to toggle labels")
    print("Press 'c' to toggle confidence scores")
    print("Press '+' to increase confidence threshold")
    print("Press '-' to decrease confidence threshold")
    if args.source:
        print(f"Source: {args.source}")
    
    # Initialize thermal capture (unified interface for live or recorded)
    print("Initializing thermal capture...")
    capture = None
    try:
        capture = ThermalCapture(args.source)
        is_recorded = capture.is_recorded
        
        if is_recorded:
            print(f"Starting human detection on pre-recorded sequence...")
            print(f"File: {args.source}")
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                print(f"Total frames: {int(total_frames)}")
            print("Press 'q' to quit, SPACE to pause/resume")
        else:
            print("Starting human detection visualization (live)...")
        
        # Download and initialize YOLO object detector with Hugging Face thermal model
        print("Downloading YOLOv8 human detection thermal model from Hugging Face...")
        if not HF_AVAILABLE:
            print("ERROR: huggingface_hub package is required for this model.")
            print("Install it with: pip install huggingface_hub")
            return
        
        model_path = hf_hub_download(
            repo_id="pitangent-ds/YOLOv8-human-detection-thermal",
            filename="model.pt"
        )
        print(f"Model downloaded to: {model_path}")
        
        print("Loading YOLO thermal detection model...")
        conf_threshold = 0.6  # Using 0.6 as in the provided example
        detector = YOLOObjectDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=0.45,
        )
        print("Model loaded successfully!")
        print(f"Available classes: {len(detector.get_class_names())}")
        
        # Visualization settings
        show_labels = True
        show_conf = True
        
        frame_count = 0  # For FPS calculation (resets every second)
        total_frame_count = 0  # Total frames processed
        fps_start_time = time.time()
        fps = 0.0
        paused = False
        last_display_time = time.time()
        
        window_name = "YOLOv8 Human Detection - Thermal Camera" + (" (Replay)" if is_recorded else "")
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
                if yuyv_frame is None:
                    if is_recorded:
                        break
                    capture.mark_frame_read()
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
                if is_recorded:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    info_lines = [
                        f"Frame: {metadata.seq} / {total_frames}",
                        f"FPS: {fps:.1f}",
                        f"Conf threshold: {conf_threshold:.2f}",
                        f"Objects detected: {len(detections)}",
                    ]
                else:
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

