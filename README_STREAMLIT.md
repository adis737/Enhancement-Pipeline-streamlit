# UDnet Underwater Image Enhancement - Streamlit App

A modern, interactive Streamlit application for underwater image enhancement using the UDnet deep learning model.

## 🌊 Features

- **Interactive Image Enhancement**: Upload and enhance underwater images with real-time processing
- **Multiple Model Support**: Choose between PyTorch (GPU/CPU) and ONNX Runtime models
- **Real-time Quality Metrics**: PSNR, SSIM, and UIQM assessment
- **Jetson Simulation**: Live performance simulation for edge deployment
- **Modern UI**: Beautiful, responsive interface with dark theme
- **Download Results**: Save enhanced images directly from the app

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Streamlit requirements
pip install -r requirements_streamlit.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For ONNX Runtime with GPU support (optional)
pip install onnxruntime-gpu
```

### 2. Prepare Model Files

Ensure you have the required model files:
- `weights/UDnet.pth` - PyTorch model weights
- `UDnet_dynamic.onnx` - ONNX model (optional, run `python export_onnx.py` to create)

### 3. Launch the App

```bash
# Simple launch
streamlit run streamlit_app.py

# Or use the launcher script
python run_streamlit.py

# Custom host and port
python run_streamlit.py --host 0.0.0.0 --port 8501
```

### 4. Open in Browser

The app will automatically open in your browser at `http://localhost:8501`

## 📱 App Interface

### Main Tabs

1. **🖼️ Image Enhancement**
   - Upload underwater images
   - Choose between PyTorch and ONNX models
   - Adjust enhancement parameters
   - View before/after comparison
   - Download enhanced results

2. **🎬 Video Processing**
   - Upload video files (coming soon)
   - Frame-by-frame enhancement
   - Real-time processing controls

3. **🚁 Jetson Demo**
   - Live performance simulation
   - Device comparison (Nano, Xavier NX, Orin NX)
   - Real-time metrics monitoring
   - Deployment readiness validation

4. **📊 Performance**
   - Model architecture overview
   - Performance characteristics
   - Deployment recommendations
   - Optimization tips

### Sidebar Controls

- **Model Selection**: Choose between PyTorch and ONNX models
- **Enhancement Parameters**: Adjust saturation and color cast neutralization
- **System Info**: View device and GPU information
- **Jetson Simulation**: Control live performance simulation

## 🔧 Configuration

### Environment Variables

```bash
# Set device preference
export CUDA_VISIBLE_DEVICES=0

# Set model paths
export UDNET_WEIGHTS_PATH=weights/UDnet.pth
export UDNET_ONNX_PATH=UDnet_dynamic.onnx
```

### Streamlit Configuration

Create `.streamlit/config.toml` for custom configuration:

```toml
[server]
port = 8501
address = "localhost"
headless = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

## 🚁 Jetson Deployment

### Supported Devices

- **Jetson Nano**: 12 FPS, 4GB GPU memory
- **Jetson Xavier NX**: 28 FPS, 8GB GPU memory
- **Jetson Orin NX**: 66 FPS, 8GB GPU memory

### Performance Optimization

1. **TensorRT Integration**: 2-3x speedup with FP16 optimization
2. **Memory Management**: Efficient GPU memory usage
3. **Real-time Processing**: Frame sampling for target FPS
4. **Edge Optimization**: Minimal resource requirements

## 📊 Quality Metrics

### Supported Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **UIQM**: Underwater Image Quality Measure

### Enhancement Parameters

- **Saturation**: Adjust color saturation (0.0 - 2.0)
- **Color Cast Neutralization**: Apply gray-world white balance
- **Model Selection**: PyTorch vs ONNX Runtime

## 🛠️ Development

### Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── run_streamlit.py          # Launcher script
├── requirements_streamlit.txt # Streamlit dependencies
├── udnet_infer.py            # Core inference module
├── model_utils/              # Model architecture
├── weights/                  # Model weights
└── UDnet_dynamic.onnx       # ONNX model
```

### Adding Features

1. **New Enhancement Options**: Add to sidebar controls
2. **Additional Metrics**: Extend `calculate_image_metrics()`
3. **Video Processing**: Implement in tab2
4. **Custom Models**: Extend model loading logic

### Testing

```bash
# Run with checks disabled
python run_streamlit.py --skip-checks

# Test with different themes
python run_streamlit.py --theme light
```

## 🚀 Deployment

### Local Deployment

```bash
# Development
streamlit run streamlit_app.py

# Production
streamlit run streamlit_app.py --server.headless true --server.port 8501
```

### Cloud Deployment

#### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with automatic updates

#### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"]
```

### Edge Deployment (Jetson)

```bash
# On Jetson device
pip install -r requirements_streamlit.txt
python run_streamlit.py --host 0.0.0.0 --port 8501
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check if `weights/UDnet.pth` exists
   - Verify PyTorch installation
   - Check GPU availability

2. **ONNX Runtime Issues**
   - Run `python export_onnx.py` to create ONNX model
   - Install `onnxruntime` or `onnxruntime-gpu`

3. **Memory Issues**
   - Reduce image size in processing
   - Use CPU mode if GPU memory is limited
   - Enable memory optimization in settings

4. **Performance Issues**
   - Use GPU acceleration when available
   - Enable TensorRT optimization on Jetson
   - Reduce input resolution for faster processing

### Debug Mode

```bash
# Enable debug logging
streamlit run streamlit_app.py --logger.level debug
```

## 📈 Performance Tips

### Optimization Strategies

1. **Model Selection**: Use PyTorch for GPU, ONNX for CPU
2. **Image Size**: Resize large images before processing
3. **Batch Processing**: Process multiple images efficiently
4. **Memory Management**: Clear cache between operations
5. **Edge Optimization**: Use TensorRT on Jetson devices

### Benchmarking

The app includes built-in performance monitoring:
- Real-time FPS tracking
- Memory usage monitoring
- Processing time measurement
- Quality metrics assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UDnet model architecture and training
- Streamlit framework for the web interface
- NVIDIA Jetson platform for edge deployment
- OpenCV and scikit-image for image processing

---

**Ready to enhance underwater images? Launch the app and dive in! 🌊🤖**
