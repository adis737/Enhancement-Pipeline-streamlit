# 🚀 Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. **Prepare Your Repository**
- Ensure your repository is public on GitHub
- Make sure `requirements.txt` is in the root directory
- Ensure `streamlit_app.py` is in the root directory

### 2. **Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `adis737/Enhancement-Pipeline-streamlit`
5. Set main file path: `streamlit_app.py`
6. Click "Deploy!"

### 3. **Troubleshooting OpenCV Issues**

If you see "OpenCV not available" error:

#### **Check Deployment Logs:**
1. Go to your app on Streamlit Cloud
2. Click the hamburger menu (☰) → "View app source"
3. Click "View logs" to see installation logs

#### **Common Issues & Solutions:**

**Issue 1: OpenCV Installation Failed**
```
ERROR: Could not find a version that satisfies the requirement opencv-python-headless
```
**Solution:** The `requirements.txt` should include:
```
opencv-python-headless>=4.4.0
```

**Issue 2: System Dependencies Missing**
```
ImportError: libGL.so.1: cannot open shared object file
```
**Solution:** Use `opencv-python-headless` instead of `opencv-python`

**Issue 3: Memory Issues**
```
DefaultCPUAllocator: not enough memory
```
**Solution:** The app now prioritizes ONNX model which uses less memory

### 4. **Verify Deployment**

After deployment, check:
- ✅ App loads without errors
- ✅ "OpenCV: ✅ Available" in System Info
- ✅ Video Processing tab shows "Video processing is available!"
- ✅ Can upload and process videos

### 5. **Debug Information**

If issues persist:
1. Go to the sidebar → "Show OpenCV Debug Info"
2. Check the debug information
3. Share the debug info for troubleshooting

## File Structure for Deployment

```
your-repo/
├── streamlit_app.py          # Main app file
├── requirements.txt          # Dependencies
├── UDnet_dynamic.onnx       # ONNX model
├── weights/
│   └── UDnet.pth            # PyTorch model
├── model_utils/             # Model utilities
└── static/                  # Output directory
```

## Requirements.txt Content

```txt
streamlit>=1.28.0
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
Pillow>=8.0.0
scikit-image>=0.17.0
scipy>=1.5.0
onnxruntime>=1.12.0
matplotlib>=3.3.0
tqdm>=4.50.0
PyYAML>=5.3.0
opencv-python-headless>=4.4.0
psutil>=5.8.0
```

## Support

If you encounter issues:
1. Check the deployment logs
2. Verify all files are in the correct location
3. Ensure the repository is public
4. Try redeploying the app

The app should work with both image and video processing on Streamlit Cloud!