#!/usr/bin/env python3
"""
Test Video Processing Functions
==============================

Simple test to verify video processing functions work correctly.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_video_functions():
    """Test video processing functions."""
    print("Testing video processing functions...")
    
    try:
        # Import the functions
        from streamlit_app import (
            ensure_static_outputs,
            choose_frame_step,
            prep_frame_for_model
        )
        
        # Test ensure_static_outputs
        print("Testing ensure_static_outputs...")
        out_dir = ensure_static_outputs()
        assert os.path.exists(out_dir), "Output directory not created"
        print(f"  Output directory: {out_dir}")
        
        # Test choose_frame_step
        print("Testing choose_frame_step...")
        step1 = choose_frame_step(30.0, 15.0)
        assert step1 == 2, f"Expected step=2, got {step1}"
        print(f"  30 FPS -> 15 FPS: step={step1}")
        
        step2 = choose_frame_step(60.0, 15.0)
        assert step2 == 4, f"Expected step=4, got {step2}"
        print(f"  60 FPS -> 15 FPS: step={step2}")
        
        # Test prep_frame_for_model
        print("Testing prep_frame_for_model...")
        test_image = Image.new('RGB', (100, 100), color='red')
        processed = prep_frame_for_model(test_image, 512)
        print(f"  Original: {test_image.size}, Processed: {processed.size}")
        
        # Test with non-multiple of 8
        test_image2 = Image.new('RGB', (100, 100), color='blue')
        processed2 = prep_frame_for_model(test_image2, 64)
        print(f"  Non-multiple of 8: {test_image2.size} -> {processed2.size}")
        
        print("\nSUCCESS: All video processing functions work correctly!")
        return True
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Test error: {e}")
        return False

def test_opencv_availability():
    """Test OpenCV availability."""
    print("\nTesting OpenCV availability...")
    
    try:
        import cv2
        print("SUCCESS: OpenCV is available")
        print(f"  Version: {cv2.__version__}")
        
        # Test basic OpenCV functions
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_array, cv2.COLOR_RGB2GRAY)
        print("SUCCESS: OpenCV basic functions work")
        
        return True
    except ImportError:
        print("WARNING: OpenCV not available - video processing will be limited")
        return False
    except Exception as e:
        print(f"ERROR: OpenCV error: {e}")
        return False

def main():
    """Main test function."""
    print("Video Processing Test Suite")
    print("=" * 40)
    
    # Test basic functions
    functions_ok = test_video_functions()
    
    # Test OpenCV
    opencv_ok = test_opencv_availability()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"  Video Functions: {'PASS' if functions_ok else 'FAIL'}")
    print(f"  OpenCV Support: {'AVAILABLE' if opencv_ok else 'NOT AVAILABLE'}")
    
    if functions_ok and opencv_ok:
        print("\nSUCCESS: Video processing is fully functional!")
    elif functions_ok:
        print("\nWARNING: Video processing functions work, but OpenCV is needed for full functionality")
    else:
        print("\nERROR: Video processing has issues")
    
    return functions_ok and opencv_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
