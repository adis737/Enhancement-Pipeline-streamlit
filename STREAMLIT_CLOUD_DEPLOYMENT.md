# Streamlit Cloud Deployment Guide

## 🚀 Quick Deploy to Streamlit Cloud

### 1. Prepare Your Repository

Make sure your GitHub repository has these files:
```
├── streamlit_app.py              # Main Streamlit app
├── requirements_streamlit_cloud.txt  # Cloud-optimized requirements
├── weights/
│   └── UDnet.pth                # Model weights (upload separately)
├── UDnet_dynamic.onnx           # ONNX model (optional)
└── udnet_infer.py               # Core inference module
```

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Connect your GitHub account**
3. **Select your repository**
4. **Choose the main file**: `streamlit_app.py`
5. **Set requirements file**: `requirements_streamlit_cloud.txt`
6. **Click Deploy!**

### 3. Upload Model Files

Since model files are large, you'll need to upload them separately:

#### Option A: GitHub LFS (Recommended)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.onnx"

# Add and commit
git add .gitattributes
git add weights/UDnet.pth
git add UDnet_dynamic.onnx
git commit -m "Add model files with LFS"
git push
```

#### Option B: External Storage
- Upload models to Google Drive, Dropbox, or AWS S3
- Update the app to download models on first run
- Add download logic to `load_models()` function

#### Option C: Streamlit Secrets
For small models, you can use Streamlit secrets:
```python
# In .streamlit/secrets.toml
[models]
model_url = "https://your-storage.com/UDnet.pth"
```

## 🔧 Cloud-Optimized Features

### OpenCV Fallback
The app automatically detects if OpenCV is available:
- ✅ **With OpenCV**: Full UIQM calculation with LAB color space
- ⚠️ **Without OpenCV**: Fallback UIQM calculation using NumPy

### Memory Optimization
- Automatic image resizing for large files
- Efficient model loading with caching
- Memory cleanup after processing

### Error Handling
- Graceful degradation when dependencies are missing
- Clear error messages for users
- Fallback functionality for all features

## 📋 Requirements for Cloud Deployment

### Core Dependencies (Always Required)
```
streamlit>=1.28.0
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
Pillow>=8.0.0
scikit-image>=0.17.0
scipy>=1.5.0
onnxruntime>=1.12.0
matplotlib>=3.3.0
```

### Optional Dependencies
```
opencv-python>=4.4.0  # May cause issues in cloud
```

## 🚨 Common Cloud Deployment Issues

### 1. OpenCV Import Error
**Error**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution**: Use `requirements_streamlit_cloud.txt` which excludes OpenCV

### 2. Model File Too Large
**Error**: Repository size limit exceeded

**Solutions**:
- Use Git LFS for large files
- Host models externally and download on demand
- Compress model files

### 3. Memory Issues
**Error**: Out of memory during model loading

**Solutions**:
- Use CPU-only models
- Implement model quantization
- Add memory monitoring

### 4. Slow Loading
**Issue**: App takes too long to start

**Solutions**:
- Use `@st.cache_resource` for model loading
- Implement lazy loading
- Optimize model size

## 🔧 Advanced Configuration

### Custom Streamlit Config
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

### Environment Variables
Set in Streamlit Cloud dashboard:
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
STREAMLIT_SERVER_HEADLESS=true
```

## 📊 Performance Optimization

### Model Loading
```python
@st.cache_resource
def load_models():
    # Cached model loading
    return enhancer, onnx_session
```

### Image Processing
```python
# Resize large images automatically
if max(w, h) > 1024:
    scale = 1024 / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    image = image.resize(new_size, Image.BICUBIC)
```

### Memory Management
```python
# Clear cache after processing
if 'processed_image' in st.session_state:
    del st.session_state.processed_image
    gc.collect()
```

## 🚀 Deployment Checklist

- [ ] Repository is public on GitHub
- [ ] `streamlit_app.py` is in root directory
- [ ] `requirements_streamlit_cloud.txt` is present
- [ ] Model files are uploaded (LFS or external)
- [ ] All dependencies are compatible with cloud
- [ ] App works locally with cloud requirements
- [ ] Error handling is implemented
- [ ] Memory usage is optimized

## 🔍 Testing Before Deployment

### Local Testing with Cloud Requirements
```bash
# Install cloud requirements
pip install -r requirements_streamlit_cloud.txt

# Test the app
streamlit run streamlit_app.py

# Check for any import errors
python -c "import streamlit_app"
```

### Performance Testing
```bash
# Test with large images
# Test with different model combinations
# Test memory usage
# Test error scenarios
```

## 📱 Post-Deployment

### Monitor Performance
- Check Streamlit Cloud logs
- Monitor memory usage
- Test with different devices
- Gather user feedback

### Updates
- Push changes to GitHub
- Streamlit Cloud auto-deploys
- Test new features thoroughly
- Monitor for issues

## 🆘 Troubleshooting

### App Won't Start
1. Check requirements file
2. Verify all imports work
3. Check model file paths
4. Review Streamlit Cloud logs

### Slow Performance
1. Optimize model loading
2. Reduce image sizes
3. Use caching effectively
4. Consider model quantization

### Memory Issues
1. Implement memory monitoring
2. Use smaller models
3. Add memory cleanup
4. Consider CPU-only deployment

---

**Your UDnet Enhancement Pipeline is now ready for cloud deployment! 🌊☁️**
