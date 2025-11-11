#!/usr/bin/env python3
"""
Test recording functionality
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pythermal import ThermalRecorder

def test_recording():
    """Test recording functionality"""
    print("=" * 60)
    print("Test: Thermal Recording")
    print("=" * 60)
    
    try:
        # Create recorder
        recorder = ThermalRecorder(output_dir="test_recordings", color=True)
        print("✓ ThermalRecorder created")
        
        # Start recording
        print("\nStarting recording (5 seconds)...")
        if not recorder.start():
            print("✗ Failed to start recording")
            return False
        
        print("✓ Recording started")
        
        # Record for 5 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            if recorder.record_frame():
                frame_count += 1
                if frame_count % 25 == 0:
                    print(f"  Recorded {frame_count} frames...")
            time.sleep(0.01)
        
        # Stop recording
        recorder.stop()
        
        # Check if file was created
        if hasattr(recorder, 'output_file') and recorder.output_file.exists():
            file_size = recorder.output_file.stat().st_size
            print(f"\n✓ Recording completed:")
            print(f"  - File: {recorder.output_file}")
            print(f"  - Size: {file_size / 1024:.1f} KB")
            print(f"  - Frames: {frame_count}")
            return True
        else:
            print("✗ Output file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    success = test_recording()
    sys.exit(0 if success else 1)

