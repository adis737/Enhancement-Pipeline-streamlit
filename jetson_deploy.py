#!/usr/bin/env python3
"""
NVIDIA Jetson Deployment Script for UDnet Underwater Image Enhancement
=====================================================================

This script demonstrates how to deploy the UDnet model on NVIDIA Jetson devices
with TensorRT optimization for maximum performance on edge devices.

Requirements:
- NVIDIA Jetson (Nano, Xavier, Orin)
- JetPack 5.0+ with TensorRT 8.5+
- ONNX model: UDnet_dynamic.onnx
- OpenCV, NumPy, PIL

Usage:
    python jetson_deploy.py --model UDnet_dynamic.onnx --input test.jpg
    python jetson_deploy.py --model UDnet_dynamic.onnx --video test.mp4 --fps 15
"""

import argparse
import os
import time
import sys
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install JetPack with TensorRT support.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install: pip install onnxruntime")


class JetsonUDNetInference:
    """Optimized UDnet inference for NVIDIA Jetson devices."""
    
    def __init__(self, model_path: str, use_tensorrt: bool = True, fp16: bool = True):
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE
        self.fp16 = fp16
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        if self.use_tensorrt:
            self._build_tensorrt_engine()
        else:
            self._setup_onnx_runtime()
    
    def _build_tensorrt_engine(self):
        """Build TensorRT engine from ONNX model."""
        print("🔧 Building TensorRT engine...")
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(self.model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 optimization enabled")
        
        # Build engine
        self.engine = builder.build_engine(network, config)
        if self.engine is None:
            print("❌ Failed to build TensorRT engine")
            sys.exit(1)
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate memory
        self._allocate_buffers()
        print("✅ TensorRT engine built successfully")
    
    def _setup_onnx_runtime(self):
        """Setup ONNX Runtime with CUDA provider."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        
        print("🔧 Setting up ONNX Runtime with CUDA...")
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider"
        ]
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("✅ ONNX Runtime with CUDA ready")
    
    def _allocate_buffers(self):
        """Allocate GPU memory for TensorRT inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        self.stream = cuda.Stream()
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray], 
                         target_size: Tuple[int, int] = (480, 640)) -> np.ndarray:
        """Preprocess image for model input."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Resize to target size (multiple of 8)
        w, h = target_size
        image = image.resize((w, h), Image.BICUBIC)
        
        # Convert to tensor format
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))  # HWC to CHW
        tensor = np.expand_dims(tensor, 0)  # Add batch dimension
        
        return tensor.astype(np.float32)
    
    def _postprocess_image(self, output: np.ndarray, original_size: Tuple[int, int]) -> Image.Image:
        """Postprocess model output to PIL Image."""
        # Remove batch dimension and convert CHW to HWC
        output = output[0].transpose(1, 2, 0)
        output = np.clip(output, 0, 1)
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        # Create PIL Image and resize to original size
        image = Image.fromarray(output)
        return image.resize(original_size, Image.BICUBIC)
    
    def infer_tensorrt(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer output data from GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Return output
        return self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1))
    
    def infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference."""
        return self.session.run([self.output_name], {self.input_name: input_tensor})[0]
    
    def enhance_image(self, image: Union[str, Image.Image, np.ndarray], 
                     target_size: Tuple[int, int] = (480, 640)) -> Tuple[Image.Image, float]:
        """Enhance a single image and return result with inference time."""
        original_size = None
        if isinstance(image, str):
            original_image = Image.open(image)
            original_size = original_image.size
        elif isinstance(image, Image.Image):
            original_image = image
            original_size = image.size
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image)
            original_size = image.shape[:2][::-1]  # (height, width) to (width, height)
        
        # Preprocess
        input_tensor = self._preprocess_image(image, target_size)
        
        # Run inference
        start_time = time.time()
        if self.use_tensorrt:
            output = self.infer_tensorrt(input_tensor)
        else:
            output = self.infer_onnx(input_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        result = self._postprocess_image(output, original_size)
        
        return result, inference_time
    
    def enhance_video(self, video_path: str, output_path: str, 
                     target_fps: float = 15.0, target_size: Tuple[int, int] = (480, 640)):
        """Enhance video with frame sampling for real-time performance."""
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame step for target FPS
        frame_step = max(1, int(original_fps / target_fps))
        output_fps = original_fps / frame_step
        
        print(f"📊 Original: {original_fps:.1f} FPS, {width}x{height}")
        print(f"📊 Target: {output_fps:.1f} FPS (step={frame_step})")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        total_inference_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to achieve target FPS
                if frame_count % frame_step != 0:
                    frame_count += 1
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Enhance frame
                enhanced, inference_time = self.enhance_image(frame_rgb, target_size)
                total_inference_time += inference_time
                processed_frames += 1
                
                # Convert back to BGR and resize to original size
                enhanced_bgr = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2BGR)
                enhanced_bgr = cv2.resize(enhanced_bgr, (width, height))
                
                # Write frame
                out.write(enhanced_bgr)
                
                frame_count += 1
                
                if frame_count % 30 == 0:  # Progress every 30 frames
                    avg_time = total_inference_time / processed_frames
                    print(f"📈 Processed {processed_frames} frames, avg: {avg_time*1000:.1f}ms/frame")
        
        finally:
            cap.release()
            out.release()
        
        avg_inference_time = total_inference_time / processed_frames
        print(f"✅ Video processing complete!")
        print(f"📊 Processed {processed_frames} frames in {total_inference_time:.2f}s")
        print(f"📊 Average inference time: {avg_inference_time*1000:.1f}ms/frame")
        print(f"📊 Effective FPS: {processed_frames/total_inference_time:.1f}")
        print(f"💾 Output saved to: {output_path}")


def benchmark_model(model_path: str, test_image: str, iterations: int = 10):
    """Benchmark model performance on Jetson."""
    print(" Starting performance benchmark...")
    
    # Test both TensorRT and ONNX Runtime
    for use_tensorrt, name in [(True, "TensorRT"), (False, "ONNX Runtime")]:
        if use_tensorrt and not TENSORRT_AVAILABLE:
            print(f"⏭️  Skipping {name} (not available)")
            continue
        
        print(f"\n🔧 Testing {name}...")
        try:
            inferencer = JetsonUDNetInference(model_path, use_tensorrt=use_tensorrt)
            
            # Warmup
            for _ in range(3):
                inferencer.enhance_image(test_image)
            
            # Benchmark
            times = []
            for i in range(iterations):
                _, inference_time = inferencer.enhance_image(test_image)
                times.append(inference_time)
                print(f"  Iteration {i+1}/{iterations}: {inference_time*1000:.1f}ms")
            
            # Statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"📊 {name} Results:")
            print(f"  Average: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
            print(f"  Min: {min_time*1000:.1f}ms")
            print(f"  Max: {max_time*1000:.1f}ms")
            print(f"  FPS: {1/avg_time:.1f}")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="UDnet Jetson Deployment")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--fps", type=float, default=15.0, help="Target FPS for video")
    parser.add_argument("--size", default="480x640", help="Target size (WxH)")
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT (default: auto)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    # Parse target size
    try:
        w, h = map(int, args.size.split('x'))
        target_size = (w, h)
    except ValueError:
        print("❌ Invalid size format. Use WxH (e.g., 480x640)")
        sys.exit(1)
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    # Run benchmark if requested
    if args.benchmark:
        if not args.input:
            print("❌ --input required for benchmark")
            sys.exit(1)
        benchmark_model(args.model, args.input, args.iterations)
        return
    
    # Create inferencer
    try:
        inferencer = JetsonUDNetInference(
            args.model, 
            use_tensorrt=args.tensorrt if args.tensorrt else TENSORRT_AVAILABLE
        )
    except Exception as e:
        print(f"❌ Failed to initialize inferencer: {e}")
        sys.exit(1)
    
    # Process image
    if args.input:
        print(f"🖼️  Processing image: {args.input}")
        try:
            result, inference_time = inferencer.enhance_image(args.input, target_size)
            
            output_path = args.output or "enhanced_jetson.jpg"
            result.save(output_path)
            
            print(f"✅ Enhanced image saved to: {output_path}")
            print(f"⏱️  Inference time: {inference_time*1000:.1f}ms")
            print(f" FPS: {1/inference_time:.1f}")
            
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            sys.exit(1)
    
    # Process video
    elif args.video:
        output_path = args.output or "enhanced_jetson.mp4"
        try:
            inferencer.enhance_video(args.video, output_path, args.fps, target_size)
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            sys.exit(1)
    
    else:
        print("❌ Please specify --input or --video")
        sys.exit(1)


if __name__ == "__main__":
    main()
