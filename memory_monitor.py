#!/usr/bin/env python3
"""
Memory Monitor for UDnet Model Loading and Inference
===================================================

This script monitors memory usage when loading and running the UDnet models.
It tracks both PyTorch (.pth) and ONNX model performance.

Usage:
    python memory_monitor.py
"""

import os
import time
import psutil
import torch
import numpy as np
from PIL import Image
import gc
from udnet_infer import UDNetEnhancer

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    return 0

def print_memory_status(stage):
    """Print current memory status."""
    cpu_mem = get_memory_usage()
    gpu_mem = get_gpu_memory()
    print(f"[{stage}] CPU Memory: {cpu_mem:.1f} MB, GPU Memory: {gpu_mem:.1f} MB")

def monitor_pytorch_model():
    """Monitor PyTorch model loading and inference."""
    print("=" * 60)
    print("MONITORING PYTORCH MODEL (.pth)")
    print("=" * 60)
    
    # Initial memory
    print_memory_status("Initial")
    
    # Check if model file exists
    model_path = "weights/UDnet.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    # Get model file size
    model_size = os.path.getsize(model_path) / 1024 / 1024  # MB
    print(f"Model file size: {model_size:.1f} MB")
    
    # Load model
    print("\nLoading PyTorch model...")
    start_time = time.time()
    
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gpu_mode = device.startswith("cuda")
        
        enhancer = UDNetEnhancer(
            weights_path=model_path,
            device=device,
            gpu_mode=gpu_mode
        )
        
        load_time = time.time() - start_time
        print_memory_status("After Model Load")
        print(f"⏱️  Model loading time: {load_time:.2f} seconds")
        
        # Create test image
        print("\n🖼️  Creating test image...")
        test_image = Image.new('RGB', (256, 256), color='blue')
        print_memory_status("After Test Image")
        
        # Run inference
        print("\n🚀 Running inference...")
        start_time = time.time()
        
        enhanced_image = enhancer.enhance_image(
            test_image,
            max_side=256,
            neutralize_cast=True,
            saturation=1.2
        )
        
        inference_time = time.time() - start_time
        print_memory_status("After Inference")
        print(f"⏱️  Inference time: {inference_time:.3f} seconds")
        
        # Memory cleanup
        print("\n🧹 Cleaning up...")
        del enhancer
        del test_image
        del enhanced_image
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print_memory_status("After Cleanup")
        
        return {
            "model_size_mb": model_size,
            "load_time": load_time,
            "inference_time": inference_time,
            "device": device,
            "gpu_mode": gpu_mode
        }
        
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        return None

def monitor_onnx_model():
    """Monitor ONNX model loading and inference."""
    print("\n" + "=" * 60)
    print("MONITORING ONNX MODEL (.onnx)")
    print("=" * 60)
    
    # Check if ONNX model exists
    onnx_path = "UDnet_dynamic.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model file not found: {onnx_path}")
        return None
    
    # Get model file size
    model_size = os.path.getsize(onnx_path) / 1024 / 1024  # MB
    print(f"📁 ONNX model file size: {model_size:.1f} MB")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime not installed")
        return None
    
    # Load ONNX model
    print("\n🔄 Loading ONNX model...")
    start_time = time.time()
    
    try:
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        load_time = time.time() - start_time
        print_memory_status("After ONNX Load")
        print(f"⏱️  ONNX loading time: {load_time:.2f} seconds")
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"📝 Input name: {input_name}")
        print(f"📝 Output name: {output_name}")
        print(f"📝 Input shape: {input_shape}")
        
        # Create test input
        print("\n🖼️  Creating test input...")
        test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        print_memory_status("After Test Input")
        
        # Run inference
        print("\n🚀 Running ONNX inference...")
        start_time = time.time()
        
        output = session.run([output_name], {input_name: test_input})[0]
        
        inference_time = time.time() - start_time
        print_memory_status("After ONNX Inference")
        print(f"⏱️  ONNX inference time: {inference_time:.3f} seconds")
        
        # Memory cleanup
        print("\n🧹 Cleaning up...")
        del session
        del test_input
        del output
        gc.collect()
        
        print_memory_status("After ONNX Cleanup")
        
        return {
            "model_size_mb": model_size,
            "load_time": load_time,
            "inference_time": inference_time,
            "input_shape": input_shape
        }
        
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return None

def main():
    """Main monitoring function."""
    print("UDnet Model Memory Monitor")
    print("=" * 60)
    
    # System info
    print(f"System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
    print(f"Python: {torch.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Monitor PyTorch model
    pytorch_results = monitor_pytorch_model()
    
    # Monitor ONNX model
    onnx_results = monitor_onnx_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if pytorch_results:
        print(f"📊 PyTorch Model:")
        print(f"   Size: {pytorch_results['model_size_mb']:.1f} MB")
        print(f"   Load Time: {pytorch_results['load_time']:.2f}s")
        print(f"   Inference Time: {pytorch_results['inference_time']:.3f}s")
        print(f"   Device: {pytorch_results['device']}")
        print(f"   GPU Mode: {pytorch_results['gpu_mode']}")
    
    if onnx_results:
        print(f"\n📊 ONNX Model:")
        print(f"   Size: {onnx_results['model_size_mb']:.1f} MB")
        print(f"   Load Time: {onnx_results['load_time']:.2f}s")
        print(f"   Inference Time: {onnx_results['inference_time']:.3f}s")
        print(f"   Input Shape: {onnx_results['input_shape']}")
    
    # Comparison
    if pytorch_results and onnx_results:
        print(f"\n📈 Comparison:")
        size_ratio = pytorch_results['model_size_mb'] / onnx_results['model_size_mb']
        print(f"   Size Ratio (PyTorch/ONNX): {size_ratio:.1f}x")
        
        if pytorch_results['inference_time'] > 0 and onnx_results['inference_time'] > 0:
            speed_ratio = pytorch_results['inference_time'] / onnx_results['inference_time']
            print(f"   Speed Ratio (PyTorch/ONNX): {speed_ratio:.1f}x")
    
    print("\n✅ Memory monitoring complete!")

if __name__ == "__main__":
    main()
