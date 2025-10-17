# 🚀 Enhancement Pipeline - Deployment Guide

## 📋 **Deployment Options Overview**

| Platform | Difficulty | Cost | Memory | Best For |
|----------|------------|------|--------|----------|
| **Streamlit Cloud** | ⭐ Easy | Free | 1GB | Quick demo, sharing |
| **Hugging Face Spaces** | ⭐ Easy | Free | 16GB | Open source projects |
| **Railway** | ⭐⭐ Medium | $5/month | 1GB | Production apps |
| **Render** | ⭐⭐ Medium | $7/month | 1GB | Reliable hosting |
| **Heroku** | ⭐⭐ Medium | $7/month | 512MB | Legacy platform |
| **AWS EC2** | ⭐⭐⭐ Hard | $10+/month | Variable | Full control |
| **Google Cloud Run** | ⭐⭐⭐ Hard | Pay-per-use | 8GB | Serverless |
| **Azure Container** | ⭐⭐⭐ Hard | Pay-per-use | 8GB | Enterprise |
| **Docker Local** | ⭐⭐ Medium | Free | Local | Development |
| **Jetson Device** | ⭐⭐⭐ Hard | Hardware cost | 4-8GB | Edge deployment |

---

## 🌟 **1. Streamlit Cloud (Recommended for Quick Start)**

### **Pros**
- ✅ Completely free
- ✅ Zero configuration
- ✅ Automatic deployments from GitHub
- ✅ Built-in HTTPS
- ✅ Custom domains

### **Cons**
- ❌ 1GB RAM limit
- ❌ No persistent storage
- ❌ Public repositories only

### **Steps**
1. **Push to GitHub** (already done ✅)
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect GitHub account**
4. **Select repository**: `Enhancement-Pipeline-streamlit`
5. **Main file path**: `streamlit_app.py`
6. **Requirements file**: `requirements_streamlit_cloud.txt`
7. **Deploy!**

### **Configuration**
```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

---

## 🤗 **2. Hugging Face Spaces (Best for Open Source)**

### **Pros**
- ✅ Free with 16GB RAM
- ✅ GPU support available
- ✅ Great for ML projects
- ✅ Built-in community features

### **Cons**
- ❌ Requires Hugging Face account
- ❌ Public by default

### **Steps**
1. **Create Hugging Face account**
2. **Create new Space**
3. **Select Streamlit SDK**
4. **Upload files**:
   - `streamlit_app.py`
   - `requirements_streamlit_cloud.txt`
   - `UDnet_dynamic.onnx`
   - `weights/UDnet.pth`
5. **Configure Space**:
   ```yaml
   # README.md
   ---
   title: Underwater Image Enhancement
   emoji: 🌊
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.50.0
   app_file: streamlit_app.py
   pinned: false
   ---
   ```

---

## 🚂 **3. Railway (Great for Production)**

### **Pros**
- ✅ Easy deployment
- ✅ Automatic HTTPS
- ✅ Database support
- ✅ $5/month starter plan

### **Steps**
1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   railway login
   ```
2. **Initialize project**:
   ```bash
   railway init
   railway add
   ```
3. **Configure Railway**:
   ```json
   // railway.json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "streamlit run streamlit_app.py --server.headless true --server.port $PORT",
       "healthcheckPath": "/",
       "healthcheckTimeout": 100
     }
   }
   ```
4. **Deploy**:
   ```bash
   railway up
   ```

---

## 🎨 **4. Render (Reliable Hosting)**

### **Pros**
- ✅ Reliable infrastructure
- ✅ Automatic deployments
- ✅ $7/month starter plan
- ✅ Good documentation

### **Steps**
1. **Connect GitHub to Render**
2. **Create new Web Service**
3. **Configure**:
   - **Build Command**: `pip install -r requirements_streamlit_cloud.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.headless true --server.port $PORT`
   - **Environment**: Python 3
4. **Add environment variables**:
   ```
   PORT=8501
   STREAMLIT_SERVER_HEADLESS=true
   ```

---

## 🐳 **5. Docker Deployment (Local/Cloud)**

### **Create Dockerfile**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_streamlit_cloud.txt .
RUN pip install --no-cache-dir -r requirements_streamlit_cloud.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### **Build and Run**
```bash
# Build image
docker build -t enhancement-pipeline .

# Run container
docker run -p 8501:8501 enhancement-pipeline

# Run with GPU (if available)
docker run --gpus all -p 8501:8501 enhancement-pipeline
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  enhancement-pipeline:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - ./weights:/app/weights
      - ./static:/app/static
    restart: unless-stopped
```

---

## ☁️ **6. Cloud Platform Deployments**

### **AWS EC2**
```bash
# Launch EC2 instance (t3.medium recommended)
# Install dependencies
sudo apt update
sudo apt install python3-pip git

# Clone repository
git clone https://github.com/yourusername/Enhancement-Pipeline-streamlit.git
cd Enhancement-Pipeline-streamlit

# Install requirements
pip3 install -r requirements_streamlit_cloud.txt

# Run with systemd service
sudo systemctl enable enhancement-pipeline
sudo systemctl start enhancement-pipeline
```

### **Google Cloud Run**
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/enhancement-pipeline', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/enhancement-pipeline']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'enhancement-pipeline', '--image', 'gcr.io/$PROJECT_ID/enhancement-pipeline', '--platform', 'managed', '--region', 'us-central1', '--allow-unauthenticated']
```

### **Azure Container Instances**
```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image enhancement-pipeline .

# Deploy to Container Instances
az container create \
  --resource-group myResourceGroup \
  --name enhancement-pipeline \
  --image myregistry.azurecr.io/enhancement-pipeline:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501 \
  --dns-name-label enhancement-pipeline
```

---

## 🔧 **7. Edge Deployment (Jetson Devices)**

### **Jetson Nano/Xavier NX/Orin NX**
```bash
# Flash JetPack SDK
# Install dependencies
sudo apt update
sudo apt install python3-pip

# Install PyTorch for Jetson
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip3 install -r jetson_requirements.txt

# Run the application
python3 streamlit_app.py
```

### **Docker on Jetson**
```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app
COPY . .
RUN pip install -r jetson_requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501"]
```

---

## 🚀 **8. Quick Deployment Scripts**

### **Local Development**
```bash
# run_local.sh
#!/bin/bash
echo "Starting Enhancement Pipeline locally..."
streamlit run streamlit_app.py --server.port 8501
```

### **Production Deployment**
```bash
# deploy_production.sh
#!/bin/bash
echo "Deploying Enhancement Pipeline to production..."

# Install dependencies
pip install -r requirements_streamlit_cloud.txt

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501

# Run with gunicorn for production
gunicorn --bind 0.0.0.0:8501 --workers 1 --threads 8 --timeout 0 streamlit_app:app
```

---

## 📊 **9. Performance Optimization**

### **Memory Optimization**
```python
# streamlit_app.py optimizations
import gc
import psutil

@st.cache_resource
def load_model():
    # Load model once and cache
    return load_onnx_model()

def cleanup_memory():
    """Clean up memory after processing"""
    gc.collect()
    if psutil.virtual_memory().percent > 80:
        st.warning("High memory usage detected")
```

### **Caching Strategy**
```python
# Cache expensive operations
@st.cache_data
def process_image(image_bytes):
    # Process and return result
    return enhanced_image

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_model_weights():
    # Load model weights
    return model
```

---

## 🔒 **10. Security Considerations**

### **Environment Variables**
```bash
# .env file
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_ENABLE_CORS=false
```

### **HTTPS Configuration**
```python
# streamlit_app.py
import streamlit as st

# Security headers
st.set_page_config(
    page_title="Enhancement Pipeline",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

---

## 📈 **11. Monitoring and Logging**

### **Health Check Endpoint**
```python
# Add to streamlit_app.py
import time

def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "memory_usage": psutil.virtual_memory().percent,
        "model_loaded": model is not None
    }

# Add health check route
if st.button("Health Check"):
    health = health_check()
    st.json(health)
```

### **Logging Configuration**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🎯 **12. Recommended Deployment Strategy**

### **For Different Use Cases**

#### **🆓 Free/Open Source**
1. **Streamlit Cloud** - Quick demo
2. **Hugging Face Spaces** - ML community

#### **💼 Production/Business**
1. **Railway** - $5/month, reliable
2. **Render** - $7/month, good support
3. **Docker** - Full control

#### **🏢 Enterprise**
1. **AWS EC2** - Full control
2. **Google Cloud Run** - Serverless
3. **Azure Container** - Microsoft ecosystem

#### **🔧 Edge/IoT**
1. **Jetson Devices** - AI edge computing
2. **Docker on Edge** - Containerized deployment

---

## 🚀 **Quick Start Commands**

### **Deploy to Streamlit Cloud (Easiest)**
```bash
# 1. Push to GitHub (already done)
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub and deploy!
```

### **Deploy with Docker**
```bash
# Build and run
docker build -t enhancement-pipeline .
docker run -p 8501:8501 enhancement-pipeline
```

### **Deploy to Railway**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

---

## 📞 **Support and Troubleshooting**

### **Common Issues**
1. **Memory errors** - Use ONNX model, not PyTorch
2. **Import errors** - Check requirements file
3. **Port conflicts** - Use different port
4. **Model not found** - Ensure model files are included

### **Debug Commands**
```bash
# Check memory usage
python memory_monitor.py

# Test model loading
python test_model_loading.py

# Check dependencies
pip list | grep -E "(torch|onnx|streamlit)"
```

---

**Choose the deployment method that best fits your needs! For quick demos, use Streamlit Cloud. For production, use Railway or Render. For full control, use Docker or cloud platforms.**
