#!/usr/bin/env python3
"""
Streamlit Launcher for UDnet Underwater Image Enhancement
=========================================================

This script provides an easy way to launch the Streamlit app with proper configuration.

Usage:
    python run_streamlit.py
    python run_streamlit.py --port 8501
    python run_streamlit.py --host 0.0.0.0 --port 8501
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'torchvision',
        'numpy',
        'Pillow',
        'scikit-image',
        'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements_streamlit.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist."""
    required_files = [
        'weights/UDnet.pth',
        'streamlit_app.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        if 'weights/UDnet.pth' in missing_files:
            print("\n💡 Download the UDnet model weights and place them in the weights/ directory")
        
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Launch UDnet Streamlit App")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind to (default: 8501)")
    parser.add_argument("--theme", default="dark", choices=["light", "dark"], help="Streamlit theme (default: dark)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip requirement and file checks")
    
    args = parser.parse_args()
    
    print("🌊 UDnet Underwater Image Enhancement - Streamlit App")
    print("=" * 60)
    
    # Check requirements
    if not args.skip_checks:
        print("🔍 Checking requirements...")
        if not check_requirements():
            sys.exit(1)
        print("✅ All required packages are installed")
        
        print("🔍 Checking model files...")
        if not check_model_files():
            sys.exit(1)
        print("✅ All required files are present")
    
    # Check for ONNX model
    if os.path.exists('UDnet_dynamic.onnx'):
        print("✅ ONNX model found - ONNX Runtime will be available")
    else:
        print("⚠️  ONNX model not found - only PyTorch model will be available")
        print("   Run 'python export_onnx.py' to create the ONNX model")
    
    # Launch Streamlit
    print(f"\n🚀 Launching Streamlit app...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Theme: {args.theme}")
    print(f"\n📱 Open your browser to: http://{args.host}:{args.port}")
    print("\n" + "=" * 60)
    
    # Build Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--theme.base", args.theme,
        "--theme.primaryColor", "#667eea",
        "--theme.backgroundColor", "#0e1117",
        "--theme.secondaryBackgroundColor", "#262730",
        "--theme.textColor", "#fafafa"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
