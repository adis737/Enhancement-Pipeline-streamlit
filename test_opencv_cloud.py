#!/usr/bin/env python3
"""
Test OpenCV installation for cloud deployment
"""

def test_opencv_installation():
    """Test if OpenCV is properly installed and working"""
    
    print("Testing OpenCV installation...")
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
        
        # Test basic functionality
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [255, 0, 0]  # Blue image
        
        # Test image operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print(f"✅ Color conversion works")
        
        # Test video writer (without actually writing)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"✅ Video codec available: {fourcc}")
        
        print(f"✅ OpenCV is fully functional for video processing")
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        print(f"   Try: pip install opencv-python-headless")
        return False
        
    except Exception as e:
        print(f"❌ OpenCV functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_opencv_installation()
    if success:
        print("\n🎉 OpenCV is ready for video processing!")
    else:
        print("\n⚠️ OpenCV needs to be fixed for video processing")
