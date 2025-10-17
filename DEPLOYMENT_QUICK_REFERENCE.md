# 🚀 Deployment Quick Reference

## ⚡ **One-Command Deployments**

### **🌐 Streamlit Cloud (Free, 2 minutes)**
```bash
python deploy.py --platform streamlit
# Then go to share.streamlit.io and connect GitHub
```

### **🚂 Railway (Production, $5/month)**
```bash
python deploy.py --platform railway
# Follow the interactive prompts
```

### **🐳 Docker (Local/Any Cloud)**
```bash
python deploy.py --platform docker
# Builds and runs locally, ready for any cloud
```

---

## 📊 **Platform Comparison**

| Platform | Cost | RAM | Setup Time | Best For |
|----------|------|-----|------------|----------|
| **Streamlit Cloud** | Free | 1GB | 2 min | Demos, sharing |
| **Hugging Face** | Free | 16GB | 5 min | ML community |
| **Railway** | $5/mo | 1GB | 3 min | Production |
| **Render** | $7/mo | 1GB | 5 min | Enterprise |
| **Docker** | Variable | Variable | 10 min | Full control |
| **Jetson** | Hardware | 4-8GB | 30 min | Edge AI |

---

## 🎯 **Recommended by Use Case**

### **🆓 Free Demo/Sharing**
```bash
python deploy.py --platform streamlit
```
- ✅ Completely free
- ✅ 2-minute setup
- ✅ Perfect for demos

### **💼 Production App**
```bash
python deploy.py --platform railway
```
- ✅ $5/month
- ✅ Reliable hosting
- ✅ Auto-deployments

### **🔬 ML Research/Community**
```bash
python deploy.py --platform huggingface
```
- ✅ Free with 16GB RAM
- ✅ GPU support
- ✅ Great for ML projects

### **🏢 Enterprise/Full Control**
```bash
python deploy.py --platform docker
```
- ✅ Deploy anywhere
- ✅ Full customization
- ✅ Scalable

---

## 💾 **Memory Requirements**

- **ONNX Model**: 54.4 MB total memory
- **Peak Usage**: 642.4 MB (with 512x512 images)
- **All platforms**: Sufficient memory available

---

## 🚀 **Quick Start Commands**

```bash
# Check if ready to deploy
python deploy.py --check-only

# Deploy to Streamlit Cloud (easiest)
python deploy.py --platform streamlit

# Deploy to Railway (production)
python deploy.py --platform railway

# Deploy with Docker (flexible)
python deploy.py --platform docker

# Show all options
python deploy.py
```

---

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Missing files**: Run `python deploy.py --check-only`
2. **Git not clean**: `git add . && git commit -m "Ready for deployment"`
3. **Memory errors**: Use ONNX model (already configured)
4. **Import errors**: Check `requirements_streamlit_cloud.txt`

### **Debug Commands**
```bash
# Check memory usage
python onnx_memory_monitor.py

# Test model loading
python -c "from udnet_infer import UDNetEnhancer; print('Model loads successfully')"

# Check dependencies
pip list | grep -E "(streamlit|onnx|torch)"
```

---

## 📞 **Support**

- **Documentation**: `DEPLOYMENT_GUIDE.md`
- **Memory Analysis**: `MEMORY_COMPARISON_SUMMARY.md`
- **Streamlit Guide**: `STREAMLIT_CLOUD_DEPLOYMENT.md`
- **Docker Guide**: `deploy_docker.py`

---

## 🎉 **Success Indicators**

✅ **Streamlit Cloud**: App accessible at `https://your-app.streamlit.app`
✅ **Railway**: App accessible at `https://your-app.railway.app`
✅ **Docker**: App accessible at `http://localhost:8501`
✅ **Hugging Face**: App accessible at `https://huggingface.co/spaces/yourusername/your-space`

---

**Choose your deployment method and run the command above! 🚀**
