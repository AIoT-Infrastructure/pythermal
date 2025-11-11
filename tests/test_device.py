#!/usr/bin/env python3
"""
Test script for thermal device functionality
"""

import sys
import time
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from pythermal import ThermalDevice, ThermalSharedMemory

def test_device_initialization():
    """Test 1: Device initialization"""
    print("=" * 60)
    print("Test 1: Device Initialization")
    print("=" * 60)
    
    try:
        device = ThermalDevice()
        print("✓ ThermalDevice created successfully")
        
        print("\nStarting thermal device...")
        device.start()
        print("✓ Thermal device started successfully")
        
        print("\nChecking if device is running...")
        if device.is_running():
            print("✓ Device is running")
        else:
            print("✗ Device is not running")
            return False
        
        return device
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_shared_memory(device):
    """Test 2: Shared memory access"""
    print("\n" + "=" * 60)
    print("Test 2: Shared Memory Access")
    print("=" * 60)
    
    try:
        shm = device.get_shared_memory()
        print("✓ Got shared memory reader")
        
        # Wait a bit for frames to arrive
        print("\nWaiting for frames (5 seconds)...")
        time.sleep(5)
        
        # Check for new frame
        if shm.has_new_frame():
            print("✓ New frame available")
            
            # Get metadata
            metadata = shm.get_metadata()
            if metadata:
                print(f"✓ Metadata retrieved:")
                print(f"  - Sequence: {metadata.seq}")
                print(f"  - Dimensions: {metadata.width}x{metadata.height}")
                print(f"  - Temperature range: {metadata.min_temp:.1f}°C - {metadata.max_temp:.1f}°C")
                print(f"  - Average: {metadata.avg_temp:.1f}°C")
            else:
                print("✗ Failed to get metadata")
                return False
            
            # Get YUYV frame
            yuyv = shm.get_yuyv_frame()
            if yuyv is not None:
                print(f"✓ YUYV frame retrieved: shape {yuyv.shape}")
            else:
                print("✗ Failed to get YUYV frame")
                return False
            
            # Get temperature array
            temp_array = shm.get_temperature_array()
            if temp_array is not None:
                print(f"✓ Temperature array retrieved: shape {temp_array.shape}")
                print(f"  - Min value: {temp_array.min()}")
                print(f"  - Max value: {temp_array.max()}")
            else:
                print("✗ Failed to get temperature array")
                return False
            
            # Get temperature map in Celsius
            temp_celsius = shm.get_temperature_map_celsius()
            if temp_celsius is not None:
                print(f"✓ Temperature map (Celsius) retrieved: shape {temp_celsius.shape}")
                print(f"  - Min temp: {temp_celsius.min():.1f}°C")
                print(f"  - Max temp: {temp_celsius.max():.1f}°C")
            else:
                print("✗ Failed to get temperature map")
                return False
            
            # Mark frame as read
            shm.mark_frame_read()
            print("✓ Frame marked as read")
            
            return True
        else:
            print("✗ No new frame available")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_streaming(device, duration=10):
    """Test 3: Frame streaming"""
    print("\n" + "=" * 60)
    print(f"Test 3: Frame Streaming ({duration} seconds)")
    print("=" * 60)
    
    try:
        shm = device.get_shared_memory()
        frame_count = 0
        start_time = time.time()
        
        print("\nStreaming frames...")
        while time.time() - start_time < duration:
            if shm.has_new_frame():
                metadata = shm.get_metadata()
                if metadata:
                    frame_count += 1
                    if frame_count % 10 == 0:
                        print(f"  Frame {frame_count}: seq={metadata.seq}, "
                              f"temp={metadata.min_temp:.1f}-{metadata.max_temp:.1f}°C")
                shm.mark_frame_read()
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n✓ Streamed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Thermal Device Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Device initialization
    device = test_device_initialization()
    if device is None:
        print("\n✗ Device initialization failed. Cannot continue.")
        return 1
    
    try:
        # Test 2: Shared memory access
        if not test_shared_memory(device):
            print("\n✗ Shared memory test failed.")
            return 1
        
        # Test 3: Frame streaming
        if not test_frame_streaming(device, duration=5):
            print("\n✗ Frame streaming test failed.")
            return 1
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    finally:
        print("\nCleaning up...")
        device.stop()
        print("✓ Device stopped")

if __name__ == "__main__":
    sys.exit(main())

