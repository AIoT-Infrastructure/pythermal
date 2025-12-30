#!/usr/bin/env python3
"""
Example: Silhouette View with Adaptive Human Detection

This example demonstrates how to create a black-white binary silhouette view
where detected humans appear as black silhouettes on a white background.
Uses adaptive human detection and applies edge refinement for sharper edges.
Supports both live camera feed and pre-recorded sequences using ThermalCapture.
"""

import cv2
import numpy as np
import argparse
import time
from pythermal import (
    ThermalCapture,
    detect_humans_adaptive,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from pythermal.utils import (
    estimate_environment_temperature_v1,
    estimate_body_temperature,
)
from pythermal.detections.utils import convert_to_celsius


def create_silhouette_mask(
    temp_array: np.ndarray,
    min_temp: float,
    max_temp: float,
    detected_objects: list,
    environment_temp: float = None,
    alpha_min: float = 0.4,
    alpha_max: float = 0.7,
    core_temp: float = 37.0,
    temp_margin: float = 2.0,
    min_temp_above_env: float = 2.0,
    max_temp_limit: float = 42.0,
) -> np.ndarray:
    """
    Create a binary mask for human silhouettes using adaptive detection.
    
    Args:
        temp_array: Temperature array (96x96)
        min_temp: Minimum temperature from metadata
        max_temp: Maximum temperature from metadata
        detected_objects: List of DetectedObject instances from adaptive detection
        environment_temp: Environment temperature (None to auto-estimate)
        alpha_min: Minimum alpha for detection
        alpha_max: Maximum alpha for detection
        core_temp: Core body temperature
        temp_margin: Temperature margin
        min_temp_above_env: Minimum temp above environment
        max_temp_limit: Maximum temperature limit
    
    Returns:
        Binary mask (96x96) where humans are 255 (will be inverted later)
    """
    # Convert to Celsius
    temp_celsius = convert_to_celsius(temp_array, min_temp, max_temp)
    
    # Estimate environment temperature if not provided
    if environment_temp is None:
        env_temp = estimate_environment_temperature_v1(temp_array, min_temp, max_temp)
        if env_temp is None:
            return np.zeros((TEMP_HEIGHT, TEMP_WIDTH), dtype=np.uint8)
    else:
        env_temp = environment_temp
    
    # Estimate body temperature range based on environment
    body_temp_min = estimate_body_temperature(env_temp, alpha=alpha_min, core_temp=core_temp)
    body_temp_max = estimate_body_temperature(env_temp, alpha=alpha_max, core_temp=core_temp)
    
    # Apply tighter bounds with margin
    detection_min = body_temp_min - temp_margin
    detection_max = body_temp_max + temp_margin
    
    # Ensure minimum temperature is significantly above environment
    detection_min = max(detection_min, env_temp + min_temp_above_env)
    
    # Cap maximum temperature to avoid detecting very hot objects
    detection_max = min(detection_max, max_temp_limit)
    
    # Ensure we have a valid range
    if detection_min >= detection_max:
        return np.zeros((TEMP_HEIGHT, TEMP_WIDTH), dtype=np.uint8)
    
    # Create binary mask for adaptive temperature range
    mask = ((temp_celsius >= detection_min) & (temp_celsius <= detection_max)).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter mask to only include areas where objects were detected
    # This helps remove false positives
    if detected_objects:
        # Create a refined mask based on detected object locations
        refined_mask = np.zeros_like(mask)
        
        # Find contours in the original mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Only keep contours that correspond to detected objects
        for contour in contours:
            # Get bounding box and center of contour
            x, y, w, h = cv2.boundingRect(contour)
            contour_center_x = x + w / 2.0
            contour_center_y = y + h / 2.0
            contour_area = cv2.contourArea(contour)
            
            # Check if this contour overlaps with any detected object
            for obj in detected_objects:
                # Calculate distance between contour center and object center
                distance = np.sqrt(
                    (contour_center_x - obj.center_x) ** 2 + 
                    (contour_center_y - obj.center_y) ** 2
                )
                
                # Check if contour overlaps with object bounding box
                obj_x1 = obj.center_x - obj.width / 2.0
                obj_y1 = obj.center_y - obj.height / 2.0
                obj_x2 = obj.center_x + obj.width / 2.0
                obj_y2 = obj.center_y + obj.height / 2.0
                
                # If contour center is within object bounding box or very close, include it
                max_dim = max(obj.width, obj.height)
                if (distance < max_dim * 1.2 or 
                    (obj_x1 <= contour_center_x <= obj_x2 and 
                     obj_y1 <= contour_center_y <= obj_y2)):
                    cv2.drawContours(refined_mask, [contour], -1, 255, -1)
                    break
        
        mask = refined_mask
    
    return mask


def refine_edges(mask: np.ndarray) -> np.ndarray:
    """
    Refine edges of the mask to make them sharper.
    
    Args:
        mask: Binary mask (0 or 255)
    
    Returns:
        Refined binary mask with sharper edges
    """
    # Apply edge detection
    edges = cv2.Canny(mask, 50, 150)
    
    # Dilate edges slightly to make them more visible
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
    
    # Combine original mask with refined edges
    # Use morphological gradient to enhance edges
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel_grad)
    
    # Combine: use original mask, but enhance edges with gradient
    refined = cv2.bitwise_or(mask, gradient)
    
    # Apply slight erosion then dilation to smooth and sharpen edges
    kernel_refine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined = cv2.erode(refined, kernel_refine, iterations=1)
    refined = cv2.dilate(refined, kernel_refine, iterations=1)
    
    # Apply closing to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
    
    return refined


def create_silhouette_view(
    temp_array: np.ndarray,
    detected_objects: list,
    min_temp: float,
    max_temp: float,
    environment_temp: float = None,
    alpha_min: float = 0.4,
    alpha_max: float = 0.7,
) -> np.ndarray:
    """
    Create a silhouette view: humans in black, background in white.
    
    Args:
        temp_array: Temperature array (96x96)
        detected_objects: List of DetectedObject instances
        min_temp: Minimum temperature from metadata
        max_temp: Maximum temperature from metadata
        environment_temp: Environment temperature (None to auto-estimate)
        alpha_min: Minimum alpha for detection
        alpha_max: Maximum alpha for detection
    
    Returns:
        Binary image (240x240) with humans as black (0) and background as white (255)
    """
    # Create mask for human silhouettes
    mask = create_silhouette_mask(
        temp_array=temp_array,
        min_temp=min_temp,
        max_temp=max_temp,
        detected_objects=detected_objects,
        environment_temp=environment_temp,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    
    # Refine edges for sharper silhouette
    mask_refined = refine_edges(mask)
    
    # Upscale mask to display size (96x96 -> 240x240)
    mask_upscaled = cv2.resize(
        mask_refined, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    
    # Threshold to ensure binary (some interpolation may create grayscale)
    _, mask_binary = cv2.threshold(mask_upscaled, 127, 255, cv2.THRESH_BINARY)
    
    # Invert: humans become black (0), background becomes white (255)
    silhouette = 255 - mask_binary
    
    return silhouette


def main():
    """Main function to run silhouette view on live or recorded data"""
    parser = argparse.ArgumentParser(
        description="Create silhouette view of detected humans (live or recorded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live silhouette view (default)
  python silhouette_view.py
  
  # Recorded silhouette view
  python silhouette_view.py recordings/thermal_20240101.tseq
  
  # Recorded with custom parameters
  python silhouette_view.py file.tseq --fps 30
  
  # With known room temperature
  python silhouette_view.py --env-temp 22.0
  
  # Face-only detection (warmer body parts)
  python silhouette_view.py --face-only
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
        "--face-only",
        action="store_true",
        help="Detect faces only (warmer body parts, α ≈ 0.5–0.7)"
    )
    parser.add_argument(
        "--env-temp",
        type=float,
        default=None,
        help="Environment temperature in Celsius. If not provided, will be estimated from frame."
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
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Index of the USB device to use (0 for first device, 1 for second, etc.). Default: 0"
    )
    
    args = parser.parse_args()
    
    # Initialize thermal capture (unified interface)
    print("Initializing thermal capture...")
    try:
        capture = ThermalCapture(args.source, device_index=args.device_index)
        is_recorded = capture.is_recorded
        
        if is_recorded:
            print("Starting silhouette view on pre-recorded sequence...")
            print(f"File: {args.source}")
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                print(f"Total frames: {int(total_frames)}")
            print("Press 'q' to quit, SPACE to pause/resume")
        else:
            print("Starting thermal silhouette view (live)...")
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
    window_name = "Thermal Silhouette View" + (" (Replay)" if is_recorded else "")
    
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
                metadata = capture.get_metadata()
                temp_array = capture.get_temperature_array()
                yuyv_frame = capture.get_yuyv_frame()
                
                if metadata is None or temp_array is None or yuyv_frame is None:
                    if is_recorded:
                        print(f"\nEnd of file reached. Processed {frame_count} frames")
                        break
                    continue
                
                # Detect humans using adaptive method
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
                    alpha_min = 0.5
                    alpha_max = 0.7
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
                    alpha_min = 0.4
                    alpha_max = 0.7
                
                # Estimate environment temperature for display
                env_temp_display = args.env_temp
                if env_temp_display is None:
                    env_temp_display = estimate_environment_temperature_v1(
                        temp_array, metadata.min_temp, metadata.max_temp
                    )
                
                # Create silhouette view
                silhouette = create_silhouette_view(
                    temp_array=temp_array,
                    detected_objects=objects,
                    min_temp=metadata.min_temp,
                    max_temp=metadata.max_temp,
                    environment_temp=env_temp_display,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                )
                
                # Convert to BGR for display (grayscale -> BGR)
                silhouette_bgr = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
                
                # Add info text
                if is_recorded:
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    info_text = [
                        f"Frame: {metadata.seq} / {total_frames}",
                        f"Humans detected: {len(objects)}",
                    ]
                else:
                    info_text = [
                        f"Frame: {metadata.seq}",
                        f"Humans detected: {len(objects)}",
                    ]
                
                # Add detection method info
                method_text = "Adaptive Detection"
                if args.face_only:
                    method_text += " [Face Only]"
                if env_temp_display is not None:
                    method_text += f" (Room: {env_temp_display:.1f}C)"
                info_text.append(method_text)
                
                # Draw info text (white text on dark background, or black on light)
                y_offset = 15
                for text in info_text:
                    cv2.putText(
                        silhouette_bgr,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black text (background is white)
                        1
                    )
                    y_offset += 18
                
                # Display silhouette
                cv2.imshow(window_name, silhouette_bgr)
                
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
