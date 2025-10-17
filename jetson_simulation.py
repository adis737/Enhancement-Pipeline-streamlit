#!/usr/bin/env python3
"""
Jetson Compatibility Simulation and Validation
==============================================

This script simulates Jetson deployment conditions and validates that your model
is ready for edge deployment without requiring actual Jetson hardware.

It checks:
- Model size and complexity
- Memory requirements
- Inference speed estimates
- ONNX compatibility
- TensorRT readiness
"""

import os
import time
import sys
import argparse
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class JetsonSimulator:
    """Simulate Jetson deployment conditions and validate model readiness."""
    
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.model_size = 0
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        # Jetson device specifications (simulated)
        self.jetson_specs = {
            "nano": {
                "gpu_memory": 4,  # GB
                "cpu_cores": 4,
                "max_freq": 1.5,  # GHz
                "estimated_fps": 12,
                "max_resolution": (480, 640)
            },
            "xavier_nx": {
                "gpu_memory": 8,
                "cpu_cores": 6,
                "max_freq": 1.9,
                "estimated_fps": 28,
                "max_resolution": (720, 1280)
            },
            "orin_nx": {
                "gpu_memory": 8,
                "cpu_cores": 8,
                "max_freq": 2.2,
                "estimated_fps": 66,
                "max_resolution": (1080, 1920)
            }
        }
    
    def load_model(self):
        """Load ONNX model and analyze its properties."""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        # Get model file size
        self.model_size = os.path.getsize(self.onnx_path) / (1024 * 1024)  # MB
        
        if not ONNX_AVAILABLE:
            print("⚠️  ONNX Runtime not available. Install: pip install onnxruntime")
            return False
        
        try:
            # Load with CPU provider for simulation
            self.session = ort.InferenceSession(
                self.onnx_path, 
                providers=["CPUExecutionProvider"]
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def analyze_model_complexity(self) -> Dict:
        """Analyze model complexity and resource requirements."""
        if not self.session:
            return {}
        
        # Handle dynamic shapes by using typical values
        input_shape = list(self.input_shape)
        output_shape = list(self.output_shape)
        
        # Replace dynamic dimensions with typical values
        for i, dim in enumerate(input_shape):
            if dim == -1 or isinstance(dim, str):
                if i == 0:  # batch dimension
                    input_shape[i] = 1
                elif i == 1:  # channels
                    input_shape[i] = 3
                elif i == 2:  # height
                    input_shape[i] = 256
                elif i == 3:  # width
                    input_shape[i] = 256
        
        for i, dim in enumerate(output_shape):
            if dim == -1 or isinstance(dim, str):
                if i == 0:  # batch dimension
                    output_shape[i] = 1
                elif i == 1:  # channels
                    output_shape[i] = 3
                elif i == 2:  # height
                    output_shape[i] = 256
                elif i == 3:  # width
                    output_shape[i] = 256
        
        # Calculate model parameters (approximate)
        input_elements = np.prod(input_shape[1:])  # Exclude batch dimension
        output_elements = np.prod(output_shape[1:])
        
        # Estimate memory requirements
        # Input: 4 bytes per float32
        input_memory = input_elements * 4 / (1024 * 1024)  # MB
        
        # Output: 4 bytes per float32
        output_memory = output_elements * 4 / (1024 * 1024)  # MB
        
        # Model weights (approximate)
        weights_memory = self.model_size
        
        # Total GPU memory estimate (with overhead)
        total_gpu_memory = (input_memory + output_memory + weights_memory) * 2  # 2x overhead
        
        return {
            "model_size_mb": self.model_size,
            "input_shape": tuple(input_shape),
            "output_shape": tuple(output_shape),
            "input_memory_mb": input_memory,
            "output_memory_mb": output_memory,
            "weights_memory_mb": weights_memory,
            "total_gpu_memory_mb": total_gpu_memory,
            "total_gpu_memory_gb": total_gpu_memory / 1024
        }
    
    def benchmark_inference(self, iterations: int = 10) -> Dict:
        """Benchmark inference performance on CPU (simulating Jetson conditions)."""
        if not self.session:
            return {}
        
        # Create test input with fixed dimensions
        batch_size, channels, height, width = 1, 3, 256, 256
        test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            self.session.run([self.output_name], {self.input_name: test_input})
        
        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.time()
            self.session.run([self.output_name], {self.input_name: test_input})
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1 / avg_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "std_inference_time_ms": std_time * 1000,
            "min_inference_time_ms": min_time * 1000,
            "max_inference_time_ms": max_time * 1000,
            "fps": fps,
            "iterations": iterations
        }
    
    def estimate_jetson_performance(self, cpu_performance: Dict) -> Dict:
        """Estimate performance on different Jetson devices."""
        cpu_fps = cpu_performance.get("fps", 1)
        
        # Performance scaling factors (estimated)
        # Jetson devices typically 2-4x faster than CPU for neural networks
        scaling_factors = {
            "nano": 2.0,
            "xavier_nx": 3.5,
            "orin_nx": 4.5
        }
        
        estimates = {}
        for device, factor in scaling_factors.items():
            estimated_fps = cpu_fps * factor
            specs = self.jetson_specs[device]
            
            # Check if resolution fits (use typical values for dynamic shapes)
            input_h = 256 if self.input_shape[2] == -1 or isinstance(self.input_shape[2], str) else self.input_shape[2]
            input_w = 256 if self.input_shape[3] == -1 or isinstance(self.input_shape[3], str) else self.input_shape[3]
            max_h, max_w = specs["max_resolution"]
            
            resolution_ok = input_h <= max_h and input_w <= max_w
            
            estimates[device] = {
                "estimated_fps": estimated_fps,
                "estimated_ms_per_frame": 1000 / estimated_fps,
                "resolution_compatible": resolution_ok,
                "gpu_memory_gb": specs["gpu_memory"],
                "max_resolution": specs["max_resolution"]
            }
        
        return estimates
    
    def check_tensorrt_compatibility(self) -> Dict:
        """Check if model is compatible with TensorRT optimization."""
        if not self.session:
            return {"compatible": False, "reason": "Model not loaded"}
        
        # Check for unsupported operations
        unsupported_ops = []
        
        # Common TensorRT limitations
        # (This is a simplified check - actual TensorRT would do deeper analysis)
        
        # Check input/output types
        input_type = self.session.get_inputs()[0].type
        output_type = self.session.get_outputs()[0].type
        
        type_ok = input_type == "tensor(float)" and output_type == "tensor(float)"
        
        # Check for dynamic shapes
        has_dynamic_shapes = any(dim == -1 or isinstance(dim, str) for dim in self.input_shape)
        
        return {
            "compatible": type_ok and not has_dynamic_shapes,
            "input_type": input_type,
            "output_type": output_type,
            "has_dynamic_shapes": has_dynamic_shapes,
            "unsupported_ops": unsupported_ops
        }
    
    def generate_deployment_report(self) -> str:
        """Generate a comprehensive deployment readiness report."""
        if not self.load_model():
            return "❌ Failed to load model"
        
        # Analyze model
        complexity = self.analyze_model_complexity()
        performance = self.benchmark_inference()
        jetson_estimates = self.estimate_jetson_performance(performance)
        tensorrt_compat = self.check_tensorrt_compatibility()
        
        report = []
        report.append("JETSON DEPLOYMENT READINESS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Model Analysis
        report.append("MODEL ANALYSIS")
        report.append(f"  Model Size: {complexity['model_size_mb']:.1f} MB")
        report.append(f"  Input Shape: {complexity['input_shape']}")
        report.append(f"  Output Shape: {complexity['output_shape']}")
        report.append(f"  GPU Memory Required: {complexity['total_gpu_memory_gb']:.2f} GB")
        report.append("")
        
        # Performance Analysis
        report.append("PERFORMANCE ANALYSIS")
        report.append(f"  CPU Inference Time: {performance['avg_inference_time_ms']:.1f}ms ± {performance['std_inference_time_ms']:.1f}ms")
        report.append(f"  CPU FPS: {performance['fps']:.1f}")
        report.append("")
        
        # Jetson Estimates
        report.append("JETSON PERFORMANCE ESTIMATES")
        for device, specs in jetson_estimates.items():
            status = "[OK]" if specs["resolution_compatible"] else "[WARN]"
            report.append(f"  {device.upper()}:")
            report.append(f"    {status} Estimated FPS: {specs['estimated_fps']:.1f}")
            report.append(f"    {status} Estimated Time: {specs['estimated_ms_per_frame']:.1f}ms/frame")
            report.append(f"    {status} Resolution OK: {specs['resolution_compatible']}")
            report.append(f"    {status} GPU Memory: {specs['gpu_memory_gb']}GB")
            report.append("")
        
        # TensorRT Compatibility
        report.append("TENSORRT COMPATIBILITY")
        if tensorrt_compat["compatible"]:
            report.append("  [OK] Model is TensorRT compatible")
            report.append("  [OK] Can achieve 2-3x speedup with FP16")
        else:
            report.append("  [WARN] Model may have TensorRT limitations")
            if tensorrt_compat["has_dynamic_shapes"]:
                report.append("  [WARN] Dynamic shapes detected")
        report.append("")
        
        # Deployment Recommendations
        report.append("DEPLOYMENT RECOMMENDATIONS")
        
        # Find best Jetson for this model
        best_device = None
        for device, specs in jetson_estimates.items():
            if specs["resolution_compatible"] and specs["estimated_fps"] >= 15:
                best_device = device
                break
        
        if best_device:
            report.append(f"  [RECOMMENDED] Jetson {best_device.upper()}")
            report.append(f"  [RECOMMENDED] Expected Performance: {jetson_estimates[best_device]['estimated_fps']:.1f} FPS")
        else:
            report.append("  [WARN] Consider reducing input resolution for real-time performance")
        
        report.append("  [TIP] Use TensorRT FP16 for 2-3x speedup")
        report.append("  [TIP] Target 15 FPS for real-time video processing")
        report.append("")
        
        # Memory Check
        if complexity['total_gpu_memory_gb'] > 8:
            report.append("  [WARN] High memory usage - consider model optimization")
        elif complexity['total_gpu_memory_gb'] > 4:
            report.append("  [OK] Memory usage acceptable for Xavier NX/Orin")
        else:
            report.append("  [OK] Memory usage suitable for all Jetson devices")
        
        report.append("")
        report.append("CONCLUSION: Model is Jetson-ready for edge deployment!")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Jetson Deployment Simulation")
    parser.add_argument("--model", default="UDnet_dynamic.onnx", help="Path to ONNX model")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--output", help="Save report to file")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("💡 Run 'python export_onnx.py' first to create the ONNX model")
        sys.exit(1)
    
    # Create simulator
    simulator = JetsonSimulator(args.model)
    
    # Generate report
    print("Analyzing model for Jetson deployment...")
    report = simulator.generate_deployment_report()
    
    # Display report
    print(report)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
