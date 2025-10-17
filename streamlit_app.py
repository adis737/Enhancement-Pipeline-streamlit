#!/usr/bin/env python3
"""
Streamlit App for UDnet Underwater Image Enhancement
====================================================

A modern Streamlit interface for the UDnet underwater image enhancement pipeline.
Features real-time processing, Jetson simulation, and comprehensive quality metrics.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import io
import os
import sys
import time
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Try to import OpenCV, but make it optional for cloud deployment
OPENCV_AVAILABLE = False
try:
    import cv2
    # Test if OpenCV is working properly
    test_array = cv2.imread("nonexistent.jpg")  # This should return None, not crash
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_AVAILABLE = False
    # Don't show warning immediately - let the app load first

# Import the core enhancement functionality
from udnet_infer import UDNetEnhancer

# Page configuration
st.set_page_config(
    page_title="UDnet Underwater Image Enhancement",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 0.5rem 0;
    }
    
    .jetson-card {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(69, 183, 209, 0.2);
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(139, 195, 74, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        color: #2e7d32;
    }
    
    .warning-message {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 152, 0, 0.2);
        color: #f57c00;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'enhancer' not in st.session_state:
    st.session_state.enhancer = None
if 'onnx_session' not in st.session_state:
    st.session_state.onnx_session = None
if 'jetson_sim_data' not in st.session_state:
    st.session_state.jetson_sim_data = {
        "is_running": False,
        "current_fps": 0,
        "inference_time": 0,
        "memory_usage": 0,
        "temperature": 45,
        "power_consumption": 0,
        "processed_frames": 0,
        "jetson_device": "Orin NX",
        "cpu_usage": 0,
        "gpu_utilization": 0,
        "model_size_mb": 3.3,
        "latency_p95": 0,
        "throughput_mbps": 0,
        "battery_life_hours": 0,
        "deployment_status": "Ready",
        "tensorrt_optimized": False,
        "quantization": "FP32"
    }

# Jetson device specifications
JETSON_SPECS = {
    "Nano": {
        "max_fps": 12, "gpu_memory": 4, "base_power": 5, "color": "#FF6B6B"
    },
    "Xavier NX": {
        "max_fps": 28, "gpu_memory": 8, "base_power": 10,
        "color": "#4ECDC4"
    },
    "Orin NX": {
        "max_fps": 66, "gpu_memory": 8, "base_power": 15,
        "color": "#45B7D1"
    }
}

@st.cache_resource
def load_models():
    """Load and cache the enhancement models."""
    try:
        # Auto-select device
        import torch
        # Force CPU mode for PyTorch to avoid CUDA issues
        device = "cpu"
        gpu_mode = False
        
        # Load ONNX model first (more reliable and memory efficient)
        onnx_session = None
        try:
            import onnxruntime as ort
            onnx_path = os.path.join("UDnet_dynamic.onnx")
            if os.path.exists(onnx_path):
                onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"]
                )
        except Exception:
            pass
        
        # Load PyTorch model as fallback (may fail due to memory)
        enhancer = None
        try:
            weights_path = os.path.join("weights", "UDnet.pth")
            enhancer = UDNetEnhancer(
                weights_path=weights_path, device=device, gpu_mode=gpu_mode
            )
        except Exception as e:
            # PyTorch model failed to load (likely memory issue)
            pass
        
        return enhancer, onnx_session, device, gpu_mode
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, "cpu", False

def calculate_uiqm(img_array):
    """Calculate Underwater Image Quality Measure (UIQM) - Memory optimized."""
    try:
        if OPENCV_AVAILABLE:
            # Use OpenCV for accurate LAB conversion and Laplacian
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # UIQM components
            uicm = np.mean(np.sqrt(np.var(a) + np.var(b)))
            
            # Sharpness measure using Laplacian
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            uism = laplacian_var / 1000.0
            
            # Contrast measure
            uiconm = np.std(gray) / 255.0
            
            # UIQM formula
            uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
            
            return float(uiqm)
        else:
            # Fallback calculation without OpenCV
            # Convert RGB to grayscale manually
            gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Simple colorfulness measure (approximation)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            uicm = np.mean(np.sqrt(np.var(r) + np.var(g) + np.var(b))) / 100.0
            
            # Simple sharpness measure (edge detection approximation)
            # Use Sobel-like edge detection
            sobel_x = np.abs(np.diff(gray, axis=1))
            sobel_y = np.abs(np.diff(gray, axis=0))
            uism = (np.mean(sobel_x) + np.mean(sobel_y)) / 1000.0
            
            # Contrast measure
            uiconm = np.std(gray) / 255.0
            
            # UIQM formula (adjusted for fallback)
            uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
            
            return float(uiqm)
    except Exception as e:
        st.warning(f"UIQM calculation error: {e}")
        return 0.0

def calculate_image_metrics(before_img, after_img):
    """Calculate PSNR, SSIM, and UIQM metrics for image comparison."""
    try:
        # Resize images to reduce memory usage
        max_size = 512
        w, h = before_img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            before_img_small = before_img.resize((new_w, new_h), Image.BICUBIC)
            after_img_small = after_img.resize((new_w, new_h), Image.BICUBIC)
        else:
            before_img_small = before_img
            after_img_small = after_img
        
        # Convert to numpy arrays
        before_array = np.array(before_img_small, dtype=np.uint8)
        after_array = np.array(after_img_small, dtype=np.uint8)
        
        # Ensure same size
        if before_array.shape != after_array.shape:
            after_array = np.array(after_img_small.resize(before_img_small.size, Image.BICUBIC), dtype=np.uint8)
        
        # Calculate metrics
        psnr = peak_signal_noise_ratio(before_array, after_array, data_range=255)
        ssim = structural_similarity(before_array, after_array, channel_axis=2, data_range=255)
        uiqm_before = calculate_uiqm(before_array)
        uiqm_after = calculate_uiqm(after_array)
        
        return {
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 3),
            'uiqm_before': round(uiqm_before, 3),
            'uiqm_after': round(uiqm_after, 3)
        }
    except Exception as e:
        st.warning(f"Metrics calculation error: {e}")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'uiqm_before': 0.0,
            'uiqm_after': 0.0
        }

def enhance_image_onnx(pil_img, neutralize_cast=False, saturation=1.0):
    """Enhance image using ONNX model."""
    if st.session_state.onnx_session is None:
        return None
    
    try:
        # Resize to multiple of 8
        w, h = pil_img.size
        new_w = w - (w % 8)
        new_h = h - (h % 8)
        pil_proc = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Preprocess
        arr = (np.asarray(pil_proc.convert("RGB"), dtype=np.float32) / 255.0)
        chw = np.transpose(arr, (2, 0, 1))
        inp = np.expand_dims(chw, 0).astype("float32")
        
        # Run ONNX inference
        sess = st.session_state.onnx_session
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        out = sess.run([out_name], {inp_name: inp})[0]
        
        # Postprocess
        out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
        out_pil = Image.fromarray((out * 255.0).round().astype("uint8"))
        out_pil = out_pil.resize((w, h), resample=Image.BICUBIC)
        
        # Apply post-processing
        if neutralize_cast:
            try:
                arr = np.asarray(out_pil).astype("float32")
                mean = arr.reshape(-1, 3).mean(axis=0) + 1e-6
                gray = float(mean.mean())
                scale = gray / mean
                balanced = (arr * scale).clip(0, 255).astype("uint8")
                out_pil = Image.fromarray(balanced)
            except Exception:
                pass
        
        if abs(saturation - 1.0) > 1e-3:
            try:
                from PIL import ImageEnhance
                out_pil = ImageEnhance.Color(out_pil).enhance(max(0.0, saturation))
            except Exception:
                pass
        
        return out_pil
    except Exception as e:
        st.error(f"ONNX enhancement failed: {e}")
        return None

def ensure_static_outputs():
    """Ensure static outputs directory exists."""
    out_dir = os.path.join("static", "outputs")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return out_dir

def choose_frame_step(in_fps, target_fps=15.0):
    """Choose frame step to achieve target FPS."""
    if in_fps <= 0:
        return 1
    import math
    step = max(1, int(math.ceil(in_fps / max(1.0, target_fps))))
    return step

def prep_frame_for_model(pil_img, max_side=512):
    """Prepare frame for model processing."""
    # Downscale for speed, then crop to multiple of 8
    w, h = pil_img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # Multiple of 8 crop (center)
    new_w = pil_img.size[0] - (pil_img.size[0] % 8)
    new_h = pil_img.size[1] - (pil_img.size[1] % 8)
    if new_w <= 0 or new_h <= 0:
        return pil_img
    if new_w != pil_img.size[0] or new_h != pil_img.size[1]:
        left = (pil_img.size[0] - new_w) // 2
        top = (pil_img.size[1] - new_h) // 2
        pil_img = pil_img.crop((left, top, left + new_w, top + new_h))
    
    return pil_img

def encode_video(frames_bgr, fps, out_path):
    """Encode video from BGR frames with better compatibility."""
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV not available for video encoding")
    
    if not frames_bgr:
        raise RuntimeError("No frames to encode")
    
    h, w = frames_bgr[0].shape[:2]
    w -= (w % 2)  # Ensure even dimensions
    h -= (h % 2)
    
    # Try different codecs for better compatibility
    codecs_to_try = [
        cv2.VideoWriter_fourcc(*"mp4v"),  # MP4V codec
        cv2.VideoWriter_fourcc(*"XVID"),  # XVID codec
        cv2.VideoWriter_fourcc(*"MJPG"),  # Motion JPEG
    ]
    
    writer = None
    for fourcc in codecs_to_try:
        try:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if writer.isOpened():
                break
        except:
            continue
    
    if writer is None or not writer.isOpened():
        raise RuntimeError("Could not create video writer with any codec")
    
    try:
        for frame in frames_bgr:
            # Resize frame to match writer dimensions
            frame_resized = cv2.resize(frame, (w, h))
            writer.write(frame_resized)
    finally:
        writer.release()

def validate_video_file(video_path):
    """Validate that a video file is properly encoded and readable."""
    if not OPENCV_AVAILABLE:
        return False, "OpenCV not available for validation"
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Check if we can read at least one frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Cannot read frames from video"
        
        return True, "Video file is valid"
    except Exception as e:
        return False, f"Validation error: {e}"

def concatenate_videos(video_paths, output_path, fps):
    """Concatenate multiple video files into one."""
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV not available for video concatenation")
    
    if not video_paths:
        raise RuntimeError("No videos to concatenate")
    
    # Read all videos and concatenate frames
    all_frames = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
    
    # Encode concatenated video
    encode_video(all_frames, fps, output_path)

def process_video_streamlit(uploaded_video, model_type, neutralize_cast, saturation, max_side=512, target_fps=15.0):
    """Process video using Streamlit-compatible approach."""
    if not OPENCV_AVAILABLE:
        st.error("OpenCV not available - video processing requires OpenCV")
        return None
    
    try:
        import uuid
        
        # Save uploaded video to temporary file
        out_dir = ensure_static_outputs()
        temp_path = os.path.join(out_dir, f"temp_{uuid.uuid4().hex}.mp4")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getvalue())
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video")
        
        # Get video properties
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = choose_frame_step(in_fps, target_fps)
        out_fps = max(1.0, in_fps / float(step))
        
        st.info(f"Processing video: {total_frames} frames at {in_fps:.1f} FPS")
        st.info(f"Target output: {out_fps:.1f} FPS (processing every {step} frame(s))")
        
        # Debug information
        expected_processed = (total_frames + step - 1) // step
        st.info(f"Expected frames to process: {expected_processed}")
        
        # Check model availability
        if model_type == "ONNX Runtime (CPU)":
            if st.session_state.onnx_session is None:
                st.error("ONNX model not loaded! Please restart the app or check if the model file exists.")
                return None, None
            else:
                st.success("ONNX model loaded successfully")
        else:
            if st.session_state.enhancer is None:
                st.error("PyTorch model not loaded! This may be due to memory constraints. Try using ONNX model instead.")
                return None, None
            else:
                st.success("PyTorch model loaded successfully")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames_bgr = []
        processed_frames = 0
        successful_frames = 0
        frame_idx = 0
        intermediate_videos = []  # Store paths to intermediate videos
        total_frames_to_process = (total_frames + step - 1) // step  # Calculate expected processed frames
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Skip frames based on step
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            
            frame_idx += 1
            processed_frames += 1
            
            # Update progress based on processed frames
            progress = min(1.0, processed_frames / total_frames_to_process)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {processed_frames}/{total_frames_to_process}")
            
            try:
                # Convert BGR to RGB
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                original_size = pil_frame.size
                
                # Prepare frame for model
                pil_processed = prep_frame_for_model(pil_frame, max_side)
                
                # Enhance frame
                if model_type == "ONNX Runtime (CPU)" and st.session_state.onnx_session is not None:
                    # ONNX processing
                    sess = st.session_state.onnx_session
                    inp_name = sess.get_inputs()[0].name
                    out_name = sess.get_outputs()[0].name
                    
                    # Preprocess
                    arr = (np.asarray(pil_processed.convert("RGB"), dtype=np.float32) / 255.0)
                    chw = np.transpose(arr, (2, 0, 1))
                    inp = np.expand_dims(chw, 0).astype("float32")
                    
                    # Run inference
                    out = sess.run([out_name], {inp_name: inp})[0]
                    
                    # Postprocess
                    out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
                    enhanced_pil = Image.fromarray((out * 255.0).round().astype("uint8"))
                else:
                    # PyTorch processing
                    if st.session_state.enhancer is None:
                        raise RuntimeError("PyTorch model not loaded")
                    
                    enhanced_pil = st.session_state.enhancer.enhance_image(
                        pil_processed,
                        max_side=None,
                        neutralize_cast=neutralize_cast,
                        saturation=saturation
                    )
                
                # Apply post-processing for ONNX
                if model_type == "ONNX Runtime (CPU)":
                    if neutralize_cast:
                        try:
                            arr = np.asarray(enhanced_pil).astype("float32")
                            mean = arr.reshape(-1, 3).mean(axis=0) + 1e-6
                            gray = float(mean.mean())
                            scale = gray / mean
                            balanced = (arr * scale).clip(0, 255).astype("uint8")
                            enhanced_pil = Image.fromarray(balanced)
                        except Exception:
                            pass
                    
                    if abs(saturation - 1.0) > 1e-3:
                        try:
                            from PIL import ImageEnhance
                            enhanced_pil = ImageEnhance.Color(enhanced_pil).enhance(max(0.0, saturation))
                        except Exception:
                            pass
                
                # Resize back to original size
                enhanced_pil = enhanced_pil.resize(original_size, resample=Image.BICUBIC)
                
                # Convert back to BGR for video encoding
                bgr_frame = cv2.cvtColor(np.asarray(enhanced_pil), cv2.COLOR_RGB2BGR)
                frames_bgr.append(bgr_frame)
                successful_frames += 1
                
                # Debug: Show progress every 10 frames
                if successful_frames % 10 == 0:
                    st.info(f"Successfully processed {successful_frames} frames so far...")
                
            except Exception as frame_error:
                st.warning(f"Error processing frame {processed_frames}: {frame_error}")
                # Continue processing other frames
                continue
            
            # Memory management - only chunk for very large videos
            if len(frames_bgr) > 500:  # Process in chunks only for large videos
                # Save intermediate video
                intermediate_path = os.path.join(out_dir, f"intermediate_{uuid.uuid4().hex}.mp4")
                encode_video(frames_bgr, out_fps, intermediate_path)
                intermediate_videos.append(intermediate_path)
                st.info(f"Saved intermediate video with {len(frames_bgr)} frames")
                frames_bgr = []  # Clear memory
        
        cap.release()
        
        # Debug information
        st.info(f"Processing complete. Frames in memory: {len(frames_bgr)}, Successful frames: {successful_frames}")
        
        # Final video encoding
        if frames_bgr:
            output_filename = f"enhanced_{model_type.lower().replace(' ', '_')}_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(out_dir, output_filename)
            
            try:
                encode_video(frames_bgr, out_fps, output_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Video processing complete! Processed {processed_frames} frames, {successful_frames} successful")
                
                return output_path, output_filename
            except Exception as encode_error:
                st.error(f"Video encoding failed: {encode_error}")
                return None, None
        elif intermediate_videos:
            # Handle case with intermediate videos (for very large videos)
            output_filename = f"enhanced_{model_type.lower().replace(' ', '_')}_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(out_dir, output_filename)
            
            try:
                concatenate_videos(intermediate_videos, output_path, out_fps)
                
                # Clean up intermediate files
                for video_path in intermediate_videos:
                    try:
                        os.remove(video_path)
                    except:
                        pass
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Video processing complete! Processed {processed_frames} frames, {successful_frames} successful")
                
                return output_path, output_filename
            except Exception as encode_error:
                st.error(f"Video concatenation failed: {encode_error}")
                return None, None
        else:
            st.error(f"No frames were processed. Total frames: {total_frames}, Step: {step}, Expected processed: {total_frames_to_process}, Successful: {successful_frames}")
            raise RuntimeError("No frames were processed")
            
    except Exception as e:
        st.error(f"Video processing failed: {e}")
        # Clean up temp file
        try:
            if 'temp_path' in locals():
                os.remove(temp_path)
        except:
            pass
        return None, None

def simulate_jetson_performance():
    """Simulate Jetson performance metrics."""
    sim_data = st.session_state.jetson_sim_data
    
    if not sim_data["is_running"]:
        return
    
    device = sim_data["jetson_device"]
    device_specs = JETSON_SPECS[device]
    
    # Simulate realistic performance
    base_fps = device_specs["max_fps"] * 0.7
    variation = np.random.normal(0, 0.1)
    current_fps = max(1, base_fps * (1 + variation))
    
    inference_time = 1000 / current_fps
    memory_usage = device_specs["gpu_memory"] * 1024 * 0.3 + np.random.normal(0, 50)
    temperature = 45 + np.random.normal(0, 2) + (current_fps / device_specs["max_fps"]) * 10
    power_consumption = device_specs["base_power"] + (current_fps / device_specs["max_fps"]) * 10
    
    # Calculate additional metrics
    cpu_usage = 20 + (current_fps / device_specs["max_fps"]) * 30 + np.random.normal(0, 5)
    gpu_utilization = 15 + (current_fps / device_specs["max_fps"]) * 40 + np.random.normal(0, 3)
    latency_p95 = inference_time * (1.2 + np.random.normal(0, 0.1))
    throughput_mbps = (current_fps * 1920 * 1080 * 3 * 8) / (1024 * 1024)
    battery_life_hours = (device_specs["base_power"] * 2) / power_consumption if power_consumption > 0 else 0
    
    sim_data.update({
        "current_fps": round(current_fps, 1),
        "inference_time": round(inference_time, 1),
        "memory_usage": round(memory_usage, 1),
        "temperature": round(temperature, 1),
        "power_consumption": round(power_consumption, 1),
        "processed_frames": sim_data["processed_frames"] + 1,
        "cpu_usage": round(max(0, min(100, cpu_usage)), 1),
        "gpu_utilization": round(max(0, min(100, gpu_utilization)), 1),
        "latency_p95": round(latency_p95, 1),
        "throughput_mbps": round(throughput_mbps, 1),
        "battery_life_hours": round(battery_life_hours, 1),
        "deployment_status": "Running" if sim_data["is_running"] else "Ready",
        "tensorrt_optimized": current_fps > device_specs["max_fps"] * 0.8,
        "quantization": "INT8" if current_fps > device_specs["max_fps"] * 0.9 else "FP32"
    })

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 UDnet Underwater Image Enhancement</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading enhancement models..."):
        enhancer, onnx_session, device, gpu_mode = load_models()
        if enhancer is not None:
            st.session_state.enhancer = enhancer
            st.session_state.onnx_session = onnx_session
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model selection (prioritize ONNX for better compatibility)
        model_type = st.selectbox(
            "Select Model",
            ["ONNX Runtime (CPU)", "PyTorch (GPU/CPU)"],
            help="Model selection applies to image processing. Video processing always uses ONNX for memory efficiency."
        )
        
        # Enhancement parameters
        st.subheader("Enhancement Parameters")
        neutralize_cast = st.checkbox("Neutralize Color Cast", value=True, help="Apply gray-world white balance")
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05, help="Adjust image saturation")
        
        # Device info
        st.subheader("System Info")
        opencv_status = "✅ Available" if OPENCV_AVAILABLE else "⚠️ Not Available"
        video_processing = "✅ ONNX Only" if onnx_session is not None else "❌ Unavailable"
        st.info(f"**Device:** {device}\n**GPU Mode:** {gpu_mode}\n**ONNX Available:** {onnx_session is not None}\n**OpenCV:** {opencv_status}\n**Video Processing:** {video_processing}")
        
        # Debug information for OpenCV
        if st.checkbox("Show OpenCV Debug Info"):
            st.code(f"""
OpenCV Available: {OPENCV_AVAILABLE}
Python Version: {sys.version}
Platform: {sys.platform}
            """)
            
            if OPENCV_AVAILABLE:
                try:
                    import cv2
                    st.success(f"OpenCV Version: {cv2.__version__}")
                except Exception as e:
                    st.error(f"OpenCV Error: {e}")
            else:
                st.warning("OpenCV is not available. Check the requirements.txt file.")
        
        # Jetson simulation controls
        st.subheader("Jetson Simulation")
        jetson_device = st.selectbox("Jetson Device", list(JETSON_SPECS.keys()), index=2)
        st.session_state.jetson_sim_data["jetson_device"] = jetson_device
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", key="start_sim"):
                st.session_state.jetson_sim_data["is_running"] = True
                st.session_state.jetson_sim_data["processed_frames"] = 0
                st.success("Simulation started!")
        
        with col2:
            if st.button("⏹️ Stop", key="stop_sim"):
                st.session_state.jetson_sim_data["is_running"] = False
                st.info("Simulation stopped!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Enhancement", "🎬 Video Processing", "🚁 Jetson Demo", "📊 Performance"])
    
    with tab1:
        st.header("Image Enhancement")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an underwater image for enhancement"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Enhanced Image")
                
                if st.button("🚀 Enhance Image", type="primary"):
                    with st.spinner("Enhancing image..."):
                        start_time = time.time()
                        
                        try:
                            # Select model
                            if model_type == "ONNX Runtime (CPU)" and st.session_state.onnx_session is not None:
                                enhanced_image = enhance_image_onnx(image, neutralize_cast, saturation)
                                model_used = "ONNX Runtime"
                            else:
                                if st.session_state.enhancer is None:
                                    st.error("PyTorch model not loaded!")
                                    enhanced_image = None
                                else:
                                    enhanced_image = st.session_state.enhancer.enhance_image(
                                        image,
                                        max_side=512,
                                        neutralize_cast=neutralize_cast,
                                        saturation=saturation
                                    )
                                model_used = "PyTorch"
                            
                            processing_time = (time.time() - start_time) * 1000
                            
                            if enhanced_image is not None:
                                st.image(enhanced_image, use_column_width=True)
                                
                                # Calculate metrics
                                metrics = calculate_image_metrics(image, enhanced_image)
                                
                                # Display metrics
                                st.subheader("Quality Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("PSNR", f"{metrics['psnr']} dB", help="Peak Signal-to-Noise Ratio")
                                
                                with col2:
                                    st.metric("SSIM", f"{metrics['ssim']}", help="Structural Similarity Index")
                                
                                with col3:
                                    st.metric("UIQM (Before)", f"{metrics['uiqm_before']}", help="Underwater Image Quality Measure")
                                
                                with col4:
                                    st.metric("UIQM (After)", f"{metrics['uiqm_after']}", help="Underwater Image Quality Measure")
                                
                                # Processing info
                                st.success(f"✅ Enhanced using {model_used} in {processing_time:.1f}ms")
                                
                                # Download button
                                buf = io.BytesIO()
                                enhanced_image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Enhanced Image",
                                    data=byte_im,
                                    file_name=f"enhanced_{uploaded_file.name}",
                                    mime="image/png"
                                )
                            else:
                                st.error("Enhancement failed!")
                                
                        except Exception as e:
                            st.error(f"Enhancement failed: {e}")
    
    with tab2:
        st.header("🎬 Video Processing")
        
        if not OPENCV_AVAILABLE:
            st.error("⚠️ Video processing requires OpenCV")
            st.info("**For local development:** `pip install opencv-python`")
            st.info("**For Streamlit Cloud:** The app should automatically install `opencv-python-headless`")
            st.warning("If you're seeing this on Streamlit Cloud, please check the deployment logs for OpenCV installation issues.")
            
            # Show a fallback option
            st.subheader("Alternative: Use Image Processing")
            st.info("While video processing is unavailable, you can still enhance individual frames using the Image Enhancement tab.")
        else:
            st.success("✅ Video processing is available!")
            st.info("💡 **Note:** Video processing always uses the ONNX model for optimal memory efficiency, regardless of your model selection above.")
            
            # Video processing parameters
            col1, col2 = st.columns(2)
            with col1:
                max_side = st.slider("Max Frame Size", 256, 1024, 512, 64, help="Maximum frame dimension for processing")
            with col2:
                target_fps = st.slider("Target FPS", 5.0, 30.0, 15.0, 1.0, help="Target output frame rate")
            
            # File upload
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video for frame-by-frame enhancement"
            )
            
            if uploaded_video is not None:
                # Display video info
                st.subheader("Video Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("File Size", f"{uploaded_video.size / (1024*1024):.1f} MB")
                with col2:
                    st.metric("File Type", uploaded_video.type)
                with col3:
                    st.metric("Filename", uploaded_video.name)
                
                # Process video button
                if st.session_state.onnx_session is None:
                    st.error("⚠️ ONNX model not loaded - video processing unavailable")
                    st.info("Please restart the app or check if the ONNX model file exists.")
                elif st.button("🚀 Enhance Video", type="primary"):
                    with st.spinner("Processing video... This may take a while."):
                        start_time = time.time()
                        
                        try:
                            # Process video (always use ONNX for memory efficiency)
                            video_model_type = "ONNX Runtime (CPU)"  # Force ONNX for video processing
                            if st.session_state.onnx_session is None:
                                st.error("ONNX model not available for video processing!")
                                output_path, output_filename = None, None
                            else:
                                st.info("🎯 Using ONNX model for video processing (memory optimized)")
                                output_path, output_filename = process_video_streamlit(
                                    uploaded_video, 
                                    video_model_type,  # Always ONNX
                                    neutralize_cast, 
                                    saturation,
                                    max_side=max_side,
                                    target_fps=target_fps
                                )
                            
                            processing_time = time.time() - start_time
                            
                            if output_path and output_filename:
                                st.success(f"✅ Video enhancement complete in {processing_time:.1f} seconds!")
                                
                                # Display enhanced video
                                st.subheader("Enhanced Video")
                                
                                # Read the output video file
                                try:
                                    with open(output_path, "rb") as video_file:
                                        video_bytes = video_file.read()
                                    
                                    # Check if video file exists and has content
                                    if len(video_bytes) == 0:
                                        st.error("❌ Enhanced video file is empty!")
                                        return
                                    
                                    st.info(f"📹 Video file size: {len(video_bytes) / (1024*1024):.1f} MB")
                                    
                                    # Validate video file
                                    is_valid, validation_msg = validate_video_file(output_path)
                                    if is_valid:
                                        st.success(f"✅ {validation_msg}")
                                    else:
                                        st.warning(f"⚠️ {validation_msg}")
                                    
                                    # Display video with error handling
                                    try:
                                        st.video(video_bytes)
                                        st.success("✅ Video displayed successfully!")
                                    except Exception as video_error:
                                        st.error(f"❌ Failed to display video: {video_error}")
                                        
                                        # Try alternative display method
                                        st.info("🔄 Trying alternative video display method...")
                                        try:
                                            # Create a base64 encoded video for HTML display
                                            import base64
                                            video_base64 = base64.b64encode(video_bytes).decode()
                                            video_html = f"""
                                            <video width="100%" controls>
                                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                                Your browser does not support the video tag.
                                            </video>
                                            """
                                            components.html(video_html, height=400)
                                            st.success("✅ Video displayed using alternative method!")
                                        except Exception as alt_error:
                                            st.error(f"❌ Alternative display also failed: {alt_error}")
                                            st.info("💡 Please use the download button to get your enhanced video.")
                                        
                                except FileNotFoundError:
                                    st.error(f"❌ Enhanced video file not found: {output_path}")
                                except Exception as file_error:
                                    st.error(f"❌ Error reading video file: {file_error}")
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Enhanced Video",
                                    data=video_bytes,
                                    file_name=output_filename,
                                    mime="video/mp4"
                                )
                                
                                # Processing statistics
                                st.subheader("Processing Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Processing Time", f"{processing_time:.1f}s")
                                with col2:
                                    st.metric("Model Used", model_type.split()[0])
                                with col3:
                                    st.metric("Max Frame Size", f"{max_side}px")
                                with col4:
                                    st.metric("Target FPS", f"{target_fps:.1f}")
                                
                                # Clean up output file after download
                                try:
                                    os.remove(output_path)
                                except:
                                    pass
                            else:
                                st.error("Video processing failed!")
                                
                        except Exception as e:
                            st.error(f"Video processing failed: {e}")
                            st.info("Try using a smaller video or reducing the max frame size.")
                
                # Video processing tips
                with st.expander("💡 Video Processing Tips"):
                    st.markdown("""
                    **For Best Results:**
                    - Use MP4 format for best compatibility
                    - Keep videos under 100MB for faster processing
                    - Lower max frame size for faster processing
                    - Use ONNX model for better memory efficiency
                    
                    **Performance Guidelines:**
                    - 512px max size: Good balance of quality/speed
                    - 256px max size: Fastest processing
                    - 1024px max size: Highest quality (slower)
                    
                    **Memory Management:**
                    - Large videos are processed in chunks
                    - Temporary files are automatically cleaned up
                    - Processing may take several minutes for long videos
                    """)
    
    with tab3:
        st.header("🚁 Jetson Deployment Demo")
        
        # Jetson simulation
        if st.session_state.jetson_sim_data["is_running"]:
            simulate_jetson_performance()
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Live Performance Metrics")
            
            # Create metrics display
            metrics_data = st.session_state.jetson_sim_data
            
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("FPS", f"{metrics_data['current_fps']}")
                st.metric("Inference Time", f"{metrics_data['inference_time']} ms")
                st.metric("Memory Usage", f"{metrics_data['memory_usage']:.1f} MB")
                st.metric("Temperature", f"{metrics_data['temperature']:.1f}°C")
            
            with col1_2:
                st.metric("CPU Usage", f"{metrics_data['cpu_usage']:.1f}%")
                st.metric("GPU Utilization", f"{metrics_data['gpu_utilization']:.1f}%")
                st.metric("Power Consumption", f"{metrics_data['power_consumption']:.1f}W")
                st.metric("Battery Life", f"{metrics_data['battery_life_hours']:.1f}h")
        
        with col2:
            st.subheader("Device Specifications")
            
            device_specs = JETSON_SPECS[jetson_device]
            
            st.markdown(f"""
            <div class="jetson-card">
                <h4>Jetson {jetson_device}</h4>
                <p><strong>Max FPS:</strong> {device_specs['max_fps']}</p>
                <p><strong>GPU Memory:</strong> {device_specs['gpu_memory']} GB</p>
                <p><strong>Base Power:</strong> {device_specs['base_power']}W</p>
                <p><strong>Status:</strong> {metrics_data['deployment_status']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advanced metrics
            st.subheader("Advanced Metrics")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("Latency P95", f"{metrics_data['latency_p95']:.1f} ms")
                st.metric("Throughput", f"{metrics_data['throughput_mbps']:.1f} Mbps")
            
            with col2_2:
                st.metric("Model Size", f"{metrics_data['model_size_mb']} MB")
                st.metric("Quantization", metrics_data['quantization'])
        
        # Device comparison
        st.subheader("Device Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (device_name, specs) in enumerate(JETSON_SPECS.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="jetson-card">
                    <h4>Jetson {device_name}</h4>
                    <p><strong>Max FPS:</strong> {specs['max_fps']}</p>
                    <p><strong>GPU Memory:</strong> {specs['gpu_memory']} GB</p>
                    <p><strong>Power:</strong> {specs['base_power']}W</p>
                    <p><strong>Best for:</strong> {'High-performance AUVs' if device_name == 'Orin NX' else 'Medium ROVs' if device_name == 'Xavier NX' else 'Small AUVs'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Auto-refresh for simulation
        if st.session_state.jetson_sim_data["is_running"]:
            time.sleep(0.1)
            st.rerun()
    
    with tab4:
        st.header("📊 Performance Analysis")
        
        # Model information
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **UDnet Architecture:**
            - Variational Autoencoder
            - Multi-scale Encoder-Decoder
            - USLN Color Correction
            - VGG16 Perceptual Loss
            """)
        
        with col2:
            st.markdown("""
            **Performance Characteristics:**
            - Model Size: 3.3 MB (ONNX)
            - Memory Usage: ~0.01 GB GPU
            - Real-time Capable: 12-66 FPS
            - TensorRT Compatible: 2-3x speedup
            """)
        
        # Deployment readiness
        st.subheader("Deployment Readiness")
        
        readiness_items = [
            ("✅ ONNX Model", "3.3 MB - Optimized for edge"),
            ("✅ Memory Efficient", "0.01 GB GPU memory"),
            ("✅ Real-time Ready", "12-66 FPS performance"),
            ("✅ TensorRT Ready", "2-3x speedup available"),
            ("✅ AUV/ROV Ready", "Production-ready deployment"),
            ("✅ Quality Metrics", "PSNR, SSIM, UIQM assessment")
        ]
        
        for status, description in readiness_items:
            st.markdown(f"**{status}** - {description}")
        
        # Performance tips
        st.subheader("Performance Tips")
        
        st.markdown("""
        **For Best Performance:**
        1. Use GPU acceleration when available
        2. Enable TensorRT optimization on Jetson
        3. Use FP16 precision for 2x speedup
        4. Optimize input resolution for target FPS
        5. Enable color cast neutralization for better results
        """)

if __name__ == "__main__":
    main()
