import io
import os
import base64
import time
import threading
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import gc

from udnet_infer import UDNetEnhancer


def calculate_uiqm(img_array):
    """
    Calculate Underwater Image Quality Measure (UIQM)
    Memory-optimized version
    """
    try:
        import cv2
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # UIQM components
        # 1. Colorfulness Measure (UICM)
        uicm = np.mean(np.sqrt(np.var(a) + np.var(b)))
        
        # 2. Sharpness Measure (UISM) - using Laplacian
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        uism = laplacian_var / 1000.0  # Normalize
        
        # 3. Contrast Measure (UIConM) - using standard deviation
        uiconm = np.std(gray) / 255.0
        
        # UIQM formula (weights from literature)
        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        
        # Clean up memory
        del lab, l, a, b, gray
        
        return float(uiqm)
    except Exception as e:
        print(f"UIQM calculation error: {e}")
        return 0.0


def calculate_image_metrics(before_img, after_img):
    """
    Calculate PSNR, SSIM, and UIQM metrics for image comparison
    Memory-optimized version
    """
    try:
        # Resize images to reduce memory usage (max 512px on longest side)
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
        
        # Convert to numpy arrays with uint8 to save memory
        before_array = np.array(before_img_small, dtype=np.uint8)
        after_array = np.array(after_img_small, dtype=np.uint8)
        
        # Ensure same size
        if before_array.shape != after_array.shape:
            after_array = np.array(after_img_small.resize(before_img_small.size, Image.BICUBIC), dtype=np.uint8)
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(before_array, after_array, data_range=255)
        
        # Calculate SSIM
        ssim = structural_similarity(before_array, after_array, channel_axis=2, data_range=255)
        
        # Calculate UIQM for both images (using smaller arrays)
        uiqm_before = calculate_uiqm(before_array)
        uiqm_after = calculate_uiqm(after_array)
        
        # Clean up memory
        del before_array, after_array
        if 'before_img_small' in locals() and before_img_small != before_img:
            del before_img_small, after_img_small
        
        return {
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 3),
            'uiqm_before': round(uiqm_before, 3),
            'uiqm_after': round(uiqm_after, 3)
        }
    except Exception as e:
        print(f"Metrics calculation error: {e}")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'uiqm_before': 0.0,
            'uiqm_after': 0.0
        }


def create_app(device: str = "cuda:0", gpu_mode: bool = True) -> Flask:
    app = Flask(__name__)

    # Ensure a single model instance for the app lifetime
    weights_path = os.path.join("weights", "UDnet.pth")
    app.config["UDNET_ENHANCER"] = UDNetEnhancer(
        weights_path=weights_path, device=device, gpu_mode=gpu_mode
    )

    # Jetson demo simulation data
    app.config["JETSON_SIM_DATA"] = {
        "is_running": False,
        "current_fps": 0,
        "inference_time": 0,
        "memory_usage": 0,
        "temperature": 45,
        "power_consumption": 0,
        "processed_frames": 0,
        "jetson_device": "Orin NX",
        # Additional AUV/ROV metrics
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
    app.config["JETSON_SPECS"] = {
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

    # Optional ONNX Runtime session (CPU by default)
    # Load UDnet_dynamic.onnx if present
    try:
        import onnxruntime as ort  # type: ignore
        onnx_path = os.path.join("UDnet_dynamic.onnx")
        if os.path.exists(onnx_path):
            sess = ort.InferenceSession(
                onnx_path,
                providers=[
                    "CPUExecutionProvider",
                ],
            )
            app.config["ONNX_SESSION"] = sess
            app.config["ONNX_INPUT_NAME"] = sess.get_inputs()[0].name
            app.config["ONNX_OUTPUT_NAME"] = sess.get_outputs()[0].name
        else:
            app.config["ONNX_SESSION"] = None
    except Exception:
        app.config["ONNX_SESSION"] = None

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/enhance", methods=["POST"])
    def enhance():
        if "image" not in request.files:
            return redirect(url_for("index"))
        file = request.files["image"]
        if not file or file.filename == "":
            return redirect(url_for("index"))

        try:
            # Read input image
            pil_img = Image.open(file.stream).convert("RGB")

            # Read UI options
            neutralize_cast = bool(request.form.get("neutralize_cast"))
            try:
                saturation = float(request.form.get("saturation", 1.0))
            except Exception:
                saturation = 1.0

            # Enhance
            enhancer: UDNetEnhancer = app.config["UDNET_ENHANCER"]
            # Memory-safe inference: cap the longest side to 512px to avoid OOM
            out_img = enhancer.enhance_image(
                pil_img,
                max_side=512,
                neutralize_cast=neutralize_cast,
                saturation=saturation,
            )

            # Encode both images as base64 PNGs
            def to_b64_png(img: Image.Image) -> str:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

            before_b64 = to_b64_png(pil_img)
            after_b64 = to_b64_png(out_img)

            # Calculate image quality metrics
            metrics = calculate_image_metrics(pil_img, out_img)

            # Clean up memory
            del pil_img, out_img
            gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

            return render_template(
                "index.html",
                before_b64=before_b64,
                after_b64=after_b64,
                metrics=metrics,
            )
        except Exception as e:
            print(f"CNN enhancement error: {e}")
            # Clean up memory on error
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            # Return to index with error message
            return render_template(
                "index.html",
                error_message="CNN model failed due to memory constraints. Try using a smaller image or use the ONNX model instead."
            )

    # Helpers for ONNX inference
    def _to_multiple_of_8(w: int, h: int):
        return w - (w % 8), h - (h % 8)

    def _preprocess(pil: Image.Image):
        import numpy as np  # local import
        arr = (np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0)
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, 0).astype("float32")

    def _postprocess(out):
        import numpy as np  # local import
        out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
        return Image.fromarray((out * 255.0).round().astype("uint8"))

    # ---- Video helpers ----
    def _ensure_static_outputs() -> str:
        out_dir = os.path.join("static", "outputs")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _choose_frame_step(in_fps: float, target_fps: float = 15.0) -> int:
        # Pick an integer step so that out_fps <= target_fps
        if in_fps <= 0:
            return 1
        import math  # local import
        step = max(1, int(math.ceil(in_fps / max(1.0, target_fps))))
        return step

    def _prep_frame_for_model(
        pil_img: Image.Image, max_side: int
    ) -> Image.Image:
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
            right = left + new_w
            bottom = top + new_h
            pil_img = pil_img.crop((left, top, right, bottom))
        return pil_img

    def _encode_video(frames_bgr, fps: float, out_path: str) -> None:
        # Ensure even dimensions for many codecs
        import cv2  # local import
        if not frames_bgr:
            raise RuntimeError("No frames to encode")
        h, w = frames_bgr[0].shape[:2]
        w -= (w % 2)
        h -= (h % 2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        try:
            for fr in frames_bgr:
                if fr.shape[1] != w or fr.shape[0] != h:
                    fr = cv2.resize(fr, (w, h))
                writer.write(fr)
        finally:
            writer.release()

    def _process_video(file_storage, use_onnx: bool) -> str:
        # Returns URL to the saved video under static/outputs
        import cv2  # local import
        import numpy as np  # local import
        import uuid  # local import

        # Read UI options shared with image forms
        neutralize_cast = bool(request.form.get("neutralize_cast"))
        try:
            saturation = float(request.form.get("saturation", 1.0))
        except Exception:
            saturation = 1.0
        try:
            max_side = int(request.form.get("max_side", 720))
        except Exception:
            max_side = 720

        # Save upload to a temp path for OpenCV
        out_dir = _ensure_static_outputs()
        tmp_path = os.path.join(out_dir, f"tmp_{uuid.uuid4().hex}.mp4")
        file_storage.save(tmp_path)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video")
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = _choose_frame_step(in_fps, target_fps=15.0)
        out_fps = max(1.0, in_fps / float(step))

        frames_bgr = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step != 0:
                idx += 1
                continue
            idx += 1
            # BGR to PIL RGB
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_in = _prep_frame_for_model(pil, max_side=max_side)

            # Enhance
            if use_onnx and app.config.get("ONNX_SESSION") is not None:
                sess = app.config["ONNX_SESSION"]
                inp_name = app.config["ONNX_INPUT_NAME"]
                out_name = app.config["ONNX_OUTPUT_NAME"]
                inp = _preprocess(pil_in)
                out = sess.run([out_name], {inp_name: inp})[0]
                out_pil = _postprocess(out)
            else:
                enhancer: UDNetEnhancer = app.config["UDNET_ENHANCER"]
                out_pil = enhancer.enhance_image(
                    pil_in,
                    max_side=None,
                    neutralize_cast=neutralize_cast,
                    saturation=saturation,
                )

            # Optional post-processing for ONNX path to match image route
            if use_onnx and app.config.get("ONNX_SESSION") is not None:
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
                        from PIL import ImageEnhance as _IE
                        out_pil = _IE.Color(out_pil).enhance(
                            max(0.0, saturation)
                        )
                    except Exception:
                        pass

            # Resize back to original uploaded frame size for temporal
            # consistency in the output
            out_pil = out_pil.resize(pil.size, resample=Image.BICUBIC)
            bgr = cv2.cvtColor(np.asarray(out_pil), cv2.COLOR_RGB2BGR)
            frames_bgr.append(bgr)

        cap.release()

        # Write output video
        out_name = (
            f"enhanced_onnx_{uuid.uuid4().hex}.mp4"
            if use_onnx else f"enhanced_torch_{uuid.uuid4().hex}.mp4"
        )
        out_path = os.path.join(out_dir, out_name)
        _encode_video(frames_bgr, out_fps, out_path)

        # Remove temp upload
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return url_for("static", filename=f"outputs/{out_name}")

    @app.route("/enhance_video", methods=["POST"])
    def enhance_video():
        if "video" not in request.files:
            return redirect(url_for("index"))
        file = request.files["video"]
        if not file or file.filename == "":
            return redirect(url_for("index"))

        try:
            video_url = _process_video(file, use_onnx=False)
        except Exception:
            return redirect(url_for("index"))

        return render_template("index.html", torch_video_url=video_url)

    @app.route("/enhance_video_onnx", methods=["POST"])
    def enhance_video_onnx():
        if app.config.get("ONNX_SESSION") is None:
            return redirect(url_for("index"))
        if "video" not in request.files:
            return redirect(url_for("index"))
        file = request.files["video"]
        if not file or file.filename == "":
            return redirect(url_for("index"))

        try:
            video_url = _process_video(file, use_onnx=True)
        except Exception:
            return redirect(url_for("index"))

        return render_template("index.html", onnx_video_url=video_url)

    @app.route("/enhance_onnx", methods=["POST"])
    def enhance_onnx():
        if app.config.get("ONNX_SESSION") is None:
            return redirect(url_for("index"))

        if "image" not in request.files:
            return redirect(url_for("index"))
        file = request.files["image"]
        if not file or file.filename == "":
            return redirect(url_for("index"))

        pil_img = Image.open(file.stream).convert("RGB")

        # Read UI options
        neutralize_cast = bool(request.form.get("neutralize_cast"))
        try:
            saturation = float(request.form.get("saturation", 1.0))
        except Exception:
            saturation = 1.0

        # Resize to multiple-of-8 for the network
        w, h = pil_img.size
        new_w, new_h = _to_multiple_of_8(w, h)
        pil_proc = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)

        # Run ONNX inference
        sess = app.config["ONNX_SESSION"]
        inp_name = app.config["ONNX_INPUT_NAME"]
        out_name = app.config["ONNX_OUTPUT_NAME"]
        inp = _preprocess(pil_proc)
        out = sess.run([out_name], {inp_name: inp})[0]
        out_pil = _postprocess(out)
        out_pil = out_pil.resize((w, h), resample=Image.BICUBIC)

        # Optional post-processing similar to PyTorch path
        if neutralize_cast:
            try:
                # simple gray-world balance
                import numpy as np  # local import
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
                from PIL import ImageEnhance as _IE
                out_pil = _IE.Color(out_pil).enhance(max(0.0, saturation))
            except Exception:
                pass

        def to_b64_png(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        before_b64_onnx = to_b64_png(pil_img)
        after_b64_onnx = to_b64_png(out_pil)

        # Calculate image quality metrics for ONNX
        metrics_onnx = calculate_image_metrics(pil_img, out_pil)

        return render_template(
            "index.html",
            before_b64_onnx=before_b64_onnx,
            after_b64_onnx=after_b64_onnx,
            metrics_onnx=metrics_onnx,
        )

    # Jetson Demo Routes
    @app.route("/jetson_demo")
    def jetson_demo():
        return render_template(
            "jetson_demo.html",
            jetson_specs=app.config["JETSON_SPECS"],
            current_device=app.config["JETSON_SIM_DATA"]["jetson_device"]
        )

    @app.route("/api/jetson_status")
    def get_jetson_status():
        return jsonify(app.config["JETSON_SIM_DATA"])

    @app.route("/api/start_jetson_sim", methods=["POST"])
    def start_jetson_sim():
        data = request.get_json()
        device = data.get('device', 'Orin NX')

        app.config["JETSON_SIM_DATA"].update({
            "is_running": True,
            "jetson_device": device,
            "processed_frames": 0
        })

        return jsonify({"status": "started", "device": device})

    @app.route("/api/stop_jetson_sim", methods=["POST"])
    def stop_jetson_sim():
        app.config["JETSON_SIM_DATA"]["is_running"] = False
        return jsonify({"status": "stopped"})
    
    @app.route("/enhance_jetson", methods=["POST"])
    def enhance_jetson():
        """Enhance image using ONNX model for Jetson simulation."""
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image file selected"}), 400
        
        try:
            # Load and process image
            pil_img = Image.open(file.stream).convert("RGB")
            
            # Use ONNX model if available, otherwise fall back to PyTorch
            if app.config.get("ONNX_SESSION"):
                # ONNX inference (Jetson simulation)
                w, h = pil_img.size
                new_w, new_h = _to_multiple_of_8(w, h)
                pil_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
                
                inp = _preprocess(pil_resized).astype(np.float32)
                sess = app.config["ONNX_SESSION"]
                inp_name = app.config["ONNX_INPUT_NAME"]
                out_name = app.config["ONNX_OUTPUT_NAME"]
                
                # Simulate Jetson processing time
                import time
                start_time = time.time()
                out = sess.run([out_name], {inp_name: inp})[0]
                processing_time = (time.time() - start_time) * 1000  # ms
                
                out_pil = _postprocess(out).resize((w, h), Image.BICUBIC)
                model_type = "ONNX (Jetson Simulation)"
            else:
                # Fallback to PyTorch
                enhancer = app.config["UDNET_ENHANCER"]
                out_pil = enhancer.enhance(pil_img)
                processing_time = 50.0  # Simulated time
                model_type = "PyTorch (Fallback)"
            
            # Convert to base64 for display
            def to_b64_png(pil):
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            
            before_b64 = to_b64_png(pil_img)
            after_b64 = to_b64_png(out_pil)
            
            # Update simulation data with real processing metrics
            sim_data = app.config["JETSON_SIM_DATA"]
            sim_data.update({
                "inference_time": round(processing_time, 1),
                "current_fps": round(1000 / processing_time, 1) if processing_time > 0 else 0,
                "processed_frames": sim_data["processed_frames"] + 1,
                "deployment_status": "Processing Complete"
            })
            
            return jsonify({
                "success": True,
                "before": before_b64,
                "after": after_b64,
                "processing_time": processing_time,
                "model_type": model_type,
                "fps": round(1000 / processing_time, 1) if processing_time > 0 else 0
            })
            
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    def simulate_jetson_performance():
        """Simulate Jetson performance metrics."""
        sim_data = app.config["JETSON_SIM_DATA"]
        specs = app.config["JETSON_SPECS"]

        if not sim_data["is_running"]:
            return

        device = sim_data["jetson_device"]
        device_specs = specs[device]

        # Simulate realistic performance
        base_fps = device_specs["max_fps"] * 0.7
        variation = np.random.normal(0, 0.1)
        current_fps = max(1, base_fps * (1 + variation))

        inference_time = 1000 / current_fps
        memory_usage = (
            device_specs["gpu_memory"] * 1024 * 0.3 + np.random.normal(0, 50)
        )
        temperature = (
            45 + np.random.normal(0, 2) +
            (current_fps / device_specs["max_fps"]) * 10
        )
        power_consumption = (
            device_specs["base_power"] +
            (current_fps / device_specs["max_fps"]) * 10
        )

        # Calculate additional metrics
        cpu_usage = 20 + (current_fps / device_specs["max_fps"]) * 30 + np.random.normal(0, 5)
        gpu_utilization = 15 + (current_fps / device_specs["max_fps"]) * 40 + np.random.normal(0, 3)
        latency_p95 = inference_time * (1.2 + np.random.normal(0, 0.1))
        throughput_mbps = (current_fps * 1920 * 1080 * 3 * 8) / (1024 * 1024)  # Assuming 1080p RGB
        battery_life_hours = (device_specs["base_power"] * 2) / power_consumption if power_consumption > 0 else 0
        
        sim_data.update({
            "current_fps": round(current_fps, 1),
            "inference_time": round(inference_time, 1),
            "memory_usage": round(memory_usage, 1),
            "temperature": round(temperature, 1),
            "power_consumption": round(power_consumption, 1),
            "processed_frames": sim_data["processed_frames"] + 1,
            # Additional metrics
            "cpu_usage": round(max(0, min(100, cpu_usage)), 1),
            "gpu_utilization": round(max(0, min(100, gpu_utilization)), 1),
            "latency_p95": round(latency_p95, 1),
            "throughput_mbps": round(throughput_mbps, 1),
            "battery_life_hours": round(battery_life_hours, 1),
            "deployment_status": "Running" if sim_data["is_running"] else "Ready",
            "tensorrt_optimized": current_fps > device_specs["max_fps"] * 0.8,
            "quantization": "INT8" if current_fps > device_specs["max_fps"] * 0.9 else "FP32"
        })

    def jetson_simulation_worker():
        """Background worker for Jetson simulation."""
        while True:
            try:
                simulate_jetson_performance()
            except Exception:
                pass
            time.sleep(0.1)

    # Start Jetson simulation worker
    jetson_thread = threading.Thread(
        target=jetson_simulation_worker, daemon=True
    )
    jetson_thread.start()

    return app


if __name__ == "__main__":
    # Auto-select CPU if CUDA not available
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_mode = device.startswith("cuda")

    app = create_app(device=device, gpu_mode=gpu_mode)
    app.run(host="0.0.0.0", port=5000, debug=False)
