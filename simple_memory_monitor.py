#!/usr/bin/env python3
"""
Simple Memory Monitor for UDnet Model
=====================================

Monitors memory usage when loading and running UDnet models.
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

def main():
    """Main monitoring function."""
    print("UDnet Model Memory Monitor")
    print("=" * 60)
    
    # System info
    print(f"System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n" + "=" * 60)
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
        print(f"Model loading time: {load_time:.2f} seconds")
        
        # Create test image
        print("\nCreating test image...")
        test_image = Image.new('RGB', (256, 256), color='blue')
        print_memory_status("After Test Image")
        
        # Run inference
        print("\nRunning inference...")
        start_time = time.time()
        
        enhanced_image = enhancer.enhance_image(
            test_image,
            max_side=256,
            neutralize_cast=True,
            saturation=1.2
        )
        
        inference_time = time.time() - start_time
        print_memory_status("After Inference")
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Memory cleanup
        print("\nCleaning up...")
        del enhancer
        del test_image
        del enhanced_image
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print_memory_status("After Cleanup")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"PyTorch Model Size: {model_size:.1f} MB")
        print(f"Load Time: {load_time:.2f}s")
        print(f"Inference Time: {inference_time:.3f}s")
        print(f"Device: {device}")
        print(f"GPU Mode: {gpu_mode}")
        
    except Exception as e:
        print(f"ERROR loading PyTorch model: {e}")
    
    print("\nMemory monitoring complete!")

if __name__ == "__main__":
    main()
