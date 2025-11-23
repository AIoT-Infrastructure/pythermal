#!/usr/bin/env python3
"""
Example: Synthetic Thermal Frame Generation

Demonstrates how to generate synthetic thermal frames (.tframe format) from RGB images.
This example shows the complete pipeline:
1. Load RGB image
2. Resize/crop to 240x240
3. Segment humans using YOLO
4. Estimate pose and assign temperatures
5. Export as .tframe file
"""

import argparse
import sys
from pathlib import Path

try:
    from pythermal.synthesis import SyntheticThermalGenerator
except ImportError:
    print("Error: Synthesis module requires YOLO dependencies.")
    print("Install with: pip install pythermal[yolo]")
    print("Or: pip install ultralytics>=8.0.0")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic thermal frame from RGB image"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input RGB image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .tframe file path (default: input_name.tframe)"
    )
    parser.add_argument(
        "--body-temp",
        type=float,
        default=37.0,
        help="Core body temperature in Celsius (default: 37.0)"
    )
    parser.add_argument(
        "--clothing-temp",
        type=float,
        default=28.0,
        help="Clothing temperature in Celsius (default: 28.0)"
    )
    parser.add_argument(
        "--ambient-temp",
        type=float,
        default=22.0,
        help="Ambient temperature in Celsius (default: 22.0)"
    )
    parser.add_argument(
        "--view-mode",
        type=str,
        default="temperature",
        choices=["yuyv", "temperature", "temperature_celsius"],
        help="View mode for rendered image (default: temperature)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="nano",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="YOLO object detection model size (default: nano)"
    )
    parser.add_argument(
        "--seg-model-size",
        type=str,
        default="nano",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="YOLO segmentation model size (default: nano)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".tframe")
    
    print(f"Input image: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Body temp: {args.body_temp}°C")
    print(f"Clothing temp: {args.clothing_temp}°C")
    print(f"Ambient temp: {args.ambient_temp}°C")
    print(f"View mode: {args.view_mode}")
    print(f"Model size: {args.model_size}")
    print(f"Segmentation model size: {args.seg_model_size}")
    print()
    
    # Initialize generator
    print("Initializing synthetic thermal generator...")
    generator = SyntheticThermalGenerator(
        body_temp=args.body_temp,
        clothing_temp=args.clothing_temp,
        ambient_temp=args.ambient_temp,
        model_size=args.model_size,
        seg_model_size=args.seg_model_size,
        pose_model_size=args.model_size,
    )
    
    # Generate thermal frame
    print("Processing image...")
    print("  - Resizing/cropping to 240x240...")
    print("  - Segmenting humans...")
    print("  - Estimating pose...")
    print("  - Assigning temperatures...")
    print("  - Generating thermal frame...")
    
    try:
        thermal_frame, rendered_image = generator.generate_from_image(
            str(input_path),
            output_path=str(output_path),
            view_mode=args.view_mode,
            sequence=0,
        )
        
        print()
        print("✓ Successfully generated thermal frame!")
        print(f"  Output: {output_path}")
        print(f"  Frame sequence: {thermal_frame.metadata.seq}")
        print(f"  Temperature range: {thermal_frame.metadata.min_temp:.1f}°C - {thermal_frame.metadata.max_temp:.1f}°C")
        print(f"  Average temperature: {thermal_frame.metadata.avg_temp:.1f}°C")
        print()
        print("You can view the .tframe file using:")
        print(f"  python -m pythermal.examples.live_view {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

