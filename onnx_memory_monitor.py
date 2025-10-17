#!/usr/bin/env python3
"""
ONNX Model Memory Monitor
========================

Detailed memory monitoring for ONNX model loading and inference.
"""

import os
import time
import psutil
import numpy as np
import onnxruntime as ort
from PIL import Image
import gc

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def print_memory_status(stage):
    """Print current memory status."""
    cpu_mem = get_memory_usage()
    print(f"[{stage}] CPU Memory: {cpu_mem:.1f} MB")

def monitor_onnx_detailed():
    """Detailed ONNX model monitoring."""
    print("ONNX Model Detailed Memory Analysis")
    print("=" * 50)
    
    # Initial memory
    print_memory_status("Initial")
    initial_memory = get_memory_usage()
    
    # Check ONNX model
    onnx_path = "UDnet_dynamic.onnx"
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        return
    
    # Get model file size
    model_size = os.path.getsize(onnx_path) / 1024 / 1024  # MB
    print(f"ONNX Model file size: {model_size:.1f} MB")
    
    # Load ONNX model
    print("\nLoading ONNX model...")
    start_time = time.time()
    
    try:
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        load_time = time.time() - start_time
        print_memory_status("After ONNX Load")
        after_load_memory = get_memory_usage()
        load_memory_increase = after_load_memory - initial_memory
        
        print(f"ONNX loading time: {load_time:.2f} seconds")
        print(f"Memory increase from loading: {load_memory_increase:.1f} MB")
        
        # Get model info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        print(f"Input shape: {input_shape}")
        
        # Create test input
        print("\nCreating test input (256x256)...")
        test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        input_size_mb = test_input.nbytes / 1024 / 1024
        print(f"Input tensor size: {input_size_mb:.2f} MB")
        print_memory_status("After Test Input")
        after_input_memory = get_memory_usage()
        input_memory_increase = after_input_memory - after_load_memory
        
        # Run inference
        print("\nRunning ONNX inference...")
        start_time = time.time()
        
        output = session.run([output_name], {input_name: test_input})[0]
        
        inference_time = time.time() - start_time
        output_size_mb = output.nbytes / 1024 / 1024
        print(f"Output tensor size: {output_size_mb:.2f} MB")
        print_memory_status("After ONNX Inference")
        after_inference_memory = get_memory_usage()
        inference_memory_increase = after_inference_memory - after_input_memory
        
        print(f"ONNX inference time: {inference_time:.3f} seconds")
        
        # Test with larger image
        print("\nTesting with larger image (512x512)...")
        large_input = np.random.randn(1, 3, 512, 512).astype(np.float32)
        large_input_size_mb = large_input.nbytes / 1024 / 1024
        print(f"Large input tensor size: {large_input_size_mb:.2f} MB")
        print_memory_status("After Large Input")
        
        start_time = time.time()
        large_output = session.run([output_name], {input_name: large_input})[0]
        large_inference_time = time.time() - start_time
        large_output_size_mb = large_output.nbytes / 1024 / 1024
        print(f"Large output tensor size: {large_output_size_mb:.2f} MB")
        print_memory_status("After Large Inference")
        after_large_memory = get_memory_usage()
        
        print(f"Large inference time: {large_inference_time:.3f} seconds")
        
        # Memory cleanup
        print("\nCleaning up...")
        del session
        del test_input
        del output
        del large_input
        del large_output
        gc.collect()
        
        print_memory_status("After Cleanup")
        final_memory = get_memory_usage()
        
        # Summary
        print("\n" + "=" * 50)
        print("ONNX MEMORY SUMMARY")
        print("=" * 50)
        print(f"Initial Memory: {initial_memory:.1f} MB")
        print(f"After Model Load: {after_load_memory:.1f} MB (+{load_memory_increase:.1f} MB)")
        print(f"After Small Inference: {after_inference_memory:.1f} MB (+{inference_memory_increase:.1f} MB)")
        print(f"After Large Inference: {after_large_memory:.1f} MB")
        print(f"After Cleanup: {final_memory:.1f} MB")
        print()
        print(f"Model File Size: {model_size:.1f} MB")
        print(f"Model Load Memory: {load_memory_increase:.1f} MB")
        print(f"Small Input (256x256): {input_size_mb:.2f} MB")
        print(f"Small Output (256x256): {output_size_mb:.2f} MB")
        print(f"Large Input (512x512): {large_input_size_mb:.2f} MB")
        print(f"Large Output (512x512): {large_output_size_mb:.2f} MB")
        print()
        print(f"Total Memory for ONNX: {after_load_memory:.1f} MB")
        print(f"Peak Memory Usage: {after_large_memory:.1f} MB")
        
        return {
            "model_size_mb": model_size,
            "load_memory_mb": load_memory_increase,
            "total_memory_mb": after_load_memory,
            "peak_memory_mb": after_large_memory,
            "load_time": load_time,
            "small_inference_time": inference_time,
            "large_inference_time": large_inference_time
        }
        
    except Exception as e:
        print(f"ERROR loading ONNX model: {e}")
        return None

def main():
    """Main function."""
    print("ONNX Model Memory Analysis")
    print("=" * 50)
    
    # System info
    print(f"System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Monitor ONNX model
    results = monitor_onnx_detailed()
    
    if results:
        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(f"ONNX Model Size: {results['model_size_mb']:.1f} MB")
        print(f"Memory for Model: {results['load_memory_mb']:.1f} MB")
        print(f"Total Memory Used: {results['total_memory_mb']:.1f} MB")
        print(f"Peak Memory: {results['peak_memory_mb']:.1f} MB")
        print(f"Load Time: {results['load_time']:.2f}s")
        print(f"Small Inference: {results['small_inference_time']:.3f}s")
        print(f"Large Inference: {results['large_inference_time']:.3f}s")
        
        # Memory efficiency
        memory_efficiency = results['model_size_mb'] / results['load_memory_mb']
        print(f"Memory Efficiency: {memory_efficiency:.1f}x (model size / load memory)")
    
    print("\nONNX memory analysis complete!")

if __name__ == "__main__":
    main()
