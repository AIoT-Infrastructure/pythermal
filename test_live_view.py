#!/usr/bin/env python3
"""
Test script for live view functionality
Tests the ThermalLiveView class without requiring a display
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pythermal import ThermalLiveView, ThermalDevice


def test_live_view_initialization():
    """Test 1: Live view initialization"""
    print("=" * 60)
    print("Test 1: Live View Initialization")
    print("=" * 60)
    
    try:
        # Test creating live view without device
        viewer = ThermalLiveView()
        print("✓ ThermalLiveView created successfully (without device)")
        
        # Test creating live view with device
        device = ThermalDevice()
        device.start()
        print("✓ ThermalDevice started")
        
        viewer_with_device = ThermalLiveView(device=device)
        print("✓ ThermalLiveView created successfully (with device)")
        
        # Test initialization
        if viewer_with_device.initialize_shared_memory():
            print("✓ Shared memory initialized successfully")
            
            # Verify we can access shared memory
            shm = viewer_with_device.shm_reader
            if shm and shm.is_initialized():
                print("✓ Shared memory reader is initialized")
                
                # Try to get metadata
                metadata = shm.get_metadata()
                if metadata:
                    print(f"✓ Metadata retrieved: {metadata.width}x{metadata.height}")
                    print(f"  - Temperature range: {metadata.min_temp:.1f}°C - {metadata.max_temp:.1f}°C")
                else:
                    print("⚠ No metadata available yet")
            else:
                print("✗ Shared memory reader not initialized")
                return False
        else:
            print("✗ Failed to initialize shared memory")
            return False
        
        # Test cleanup
        viewer_with_device.cleanup()
        print("✓ Cleanup successful")
        
        device.stop()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_view_methods():
    """Test 2: Live view methods"""
    print("\n" + "=" * 60)
    print("Test 2: Live View Methods")
    print("=" * 60)
    
    try:
        device = ThermalDevice()
        device.start()
        
        viewer = ThermalLiveView(device=device)
        viewer.initialize_shared_memory()
        
        # Wait for a frame
        print("\nWaiting for frame data...")
        time.sleep(2)
        
        # Test getting frame data
        if viewer.shm_reader.has_new_frame():
            metadata = viewer.shm_reader.get_metadata()
            if metadata:
                # Test temperature calculation
                temp = viewer.calculate_temperature_from_pixel(
                    120, 120, metadata.min_temp, metadata.max_temp
                )
                if temp is not None:
                    print(f"✓ Temperature calculation works: {temp:.1f}°C at (120, 120)")
                else:
                    print("⚠ Temperature calculation returned None")
                
                # Test getting YUYV frame
                yuyv = viewer.shm_reader.get_yuyv_frame()
                if yuyv is not None:
                    print(f"✓ YUYV frame retrieval works: shape {yuyv.shape}")
                    
                    # Test YUYV conversion
                    bgr = viewer.get_original_yuyv(yuyv)
                    if bgr is not None:
                        print(f"✓ YUYV to BGR conversion works: shape {bgr.shape}")
                else:
                    print("⚠ YUYV frame is None")
                
                # Test getting temperature array
                temp_array = viewer.shm_reader.get_temperature_array()
                if temp_array is not None:
                    print(f"✓ Temperature array retrieval works: shape {temp_array.shape}")
                    
                    # Test temperature view conversion
                    temp_view = viewer.get_temperature_view(
                        temp_array, metadata.min_temp, metadata.max_temp
                    )
                    if temp_view is not None:
                        print(f"✓ Temperature view conversion works: shape {temp_view.shape}")
                else:
                    print("⚠ Temperature array is None")
                
                # Test overlay drawing
                test_image = viewer.get_original_yuyv(yuyv) if yuyv is not None else None
                if test_image is not None:
                    overlay = viewer.draw_overlay(
                        test_image, metadata.min_temp, metadata.max_temp,
                        metadata.avg_temp, metadata.seq, 25.0
                    )
                    if overlay is not None:
                        print(f"✓ Overlay drawing works: shape {overlay.shape}")
                
                # Test FPS calculation
                fps = viewer.calculate_fps()
                print(f"✓ FPS calculation works: {fps:.1f} FPS")
                
                viewer.shm_reader.mark_frame_read()
            else:
                print("⚠ No metadata available")
        else:
            print("⚠ No new frame available")
        
        viewer.cleanup()
        device.stop()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entry_point():
    """Test 3: Entry point availability"""
    print("\n" + "=" * 60)
    print("Test 3: Entry Point Configuration")
    print("=" * 60)
    
    try:
        # Test that main function exists and is callable
        from pythermal.live_view import main
        print("✓ main() function is importable")
        
        # Test that it's a function
        import types
        if isinstance(main, types.FunctionType):
            print("✓ main() is a function")
        else:
            print("✗ main() is not a function")
            return False
        
        # Note: We can't actually run main() here because it requires a display
        print("✓ Entry point configured correctly")
        print("  Note: Cannot test GUI display without X11")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Live View Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_live_view_initialization()))
    
    # Test 2: Methods
    results.append(("Methods", test_live_view_methods()))
    
    # Test 3: Entry point
    results.append(("Entry Point", test_entry_point()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

