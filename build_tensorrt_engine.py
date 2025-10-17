#!/usr/bin/env python3
"""
TensorRT Engine Builder for UDnet on NVIDIA Jetson
==================================================

This script builds optimized TensorRT engines from the ONNX model
for different precision modes and input sizes.

Usage:
    python build_tensorrt_engine.py --model UDnet_dynamic.onnx
    python build_tensorrt_engine.py --model UDnet_dynamic.onnx --fp16 --size 480x640
"""

import argparse
import os
import sys

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("❌ TensorRT not available. Install JetPack with TensorRT support.")


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, 
                input_size: tuple = (1, 3, 480, 640), max_workspace: int = 1):
    """Build TensorRT engine from ONNX model."""
    
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")
    
    print(f"🔧 Building TensorRT engine from: {onnx_path}")
    print(f"📐 Input size: {input_size}")
    print(f"🎯 Precision: {'FP16' if fp16 else 'FP32'}")
    print(f"💾 Workspace: {max_workspace}GB")
    
    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            return False
    
    print("✅ ONNX model parsed successfully")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace * (1 << 30)  # Convert GB to bytes
    
    # Enable FP16 if requested and supported
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 optimization enabled")
    elif fp16:
        print("⚠️  FP16 requested but not supported on this platform")
    
    # Set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Get input name from network
    input_name = network.get_input(0).name
    print(f"📝 Input name: {input_name}")
    
    # Set dynamic shape ranges
    batch_size, channels, height, width = input_size
    
    # Minimum, optimal, and maximum shapes
    min_shape = (1, channels, height//2, width//2)
    opt_shape = (batch_size, channels, height, width)
    max_shape = (batch_size, channels, height*2, width*2)
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    print(f"📐 Dynamic shapes:")
    print(f"  Min: {min_shape}")
    print(f"  Opt: {opt_shape}")
    print(f"  Max: {max_shape}")
    
    # Build engine
    print("🔨 Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build TensorRT engine")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT engine saved to: {engine_path}")
    
    # Print engine info
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized_engine)
    print(f"📊 Engine info:")
    print(f"  Max batch size: {engine.max_batch_size}")
    print(f"  Number of bindings: {engine.num_bindings}")
    print(f"  Has implicit batch dimension: {engine.has_implicit_batch_dimension}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine for UDnet")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", help="Output engine path")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision")
    parser.add_argument("--size", default="480x640", help="Input size (WxH)")
    parser.add_argument("--workspace", type=int, default=1, help="Max workspace size (GB)")
    
    args = parser.parse_args()
    
    # Check if ONNX model exists
    if not os.path.exists(args.model):
        print(f"❌ ONNX model not found: {args.model}")
        sys.exit(1)
    
    # Determine precision
    fp16 = args.fp16 and not args.fp32
    
    # Parse input size
    try:
        w, h = map(int, args.size.split('x'))
        input_size = (1, 3, h, w)  # (batch, channels, height, width)
    except ValueError:
        print("❌ Invalid size format. Use WxH (e.g., 480x640)")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        engine_path = args.output
    else:
        precision = "fp16" if fp16 else "fp32"
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        engine_path = f"{base_name}_{precision}_{w}x{h}.trt"
    
    # Build engine
    try:
        success = build_engine(
            args.model, 
            engine_path, 
            fp16=fp16, 
            input_size=input_size,
            max_workspace=args.workspace
        )
        
        if success:
            print(f"\n🎉 TensorRT engine built successfully!")
            print(f"📁 Engine file: {engine_path}")
            print(f"💡 Use this engine with jetson_deploy.py for optimal performance")
        else:
            print("❌ Failed to build TensorRT engine")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error building engine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
