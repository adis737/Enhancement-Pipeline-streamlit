#!/usr/bin/env python3
"""
Enhancement Pipeline - Universal Deployment Script
=================================================

This script provides a unified interface for deploying the Enhancement Pipeline
to various platforms including Streamlit Cloud, Railway, Docker, and more.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print deployment banner."""
    print("="*60)
    print("   ENHANCEMENT PIPELINE - DEPLOYMENT WIZARD")
    print("="*60)
    print()

def show_deployment_options():
    """Show available deployment options."""
    print("AVAILABLE DEPLOYMENT OPTIONS:")
    print("="*50)
    print()
    print("1. Streamlit Cloud (Free, Easy)")
    print("   - Completely free")
    print("   - 1GB RAM limit")
    print("   - Automatic GitHub integration")
    print("   - Perfect for demos and sharing")
    print()
    print("2. Hugging Face Spaces (Free, ML-focused)")
    print("   - Free with 16GB RAM")
    print("   - GPU support available")
    print("   - Great for ML community")
    print("   - Built-in model sharing")
    print()
    print("3. Railway (Production, $5/month)")
    print("   - Reliable hosting")
    print("   - 1GB RAM")
    print("   - Automatic deployments")
    print("   - Good for production apps")
    print()
    print("4. Render (Production, $7/month)")
    print("   - Enterprise-grade hosting")
    print("   - 1GB RAM")
    print("   - Excellent uptime")
    print("   - Great support")
    print()
    print("5. Docker (Local/Cloud, Flexible)")
    print("   - Run anywhere")
    print("   - Full control")
    print("   - Deploy to any cloud")
    print("   - Perfect for development")
    print()
    print("6. Cloud Platforms (AWS/GCP/Azure)")
    print("   - Full control")
    print("   - Scalable")
    print("   - Enterprise features")
    print("   - Pay-per-use")
    print()
    print("7. Jetson Edge Deployment")
    print("   - AI edge computing")
    print("   - Real-time processing")
    print("   - Offline capability")
    print("   - Hardware required")
    print()

def check_prerequisites():
    """Check deployment prerequisites."""
    print("🔍 CHECKING PREREQUISITES...")
    print("="*30)
    
    # Check required files
    required_files = [
        "streamlit_app.py",
        "requirements_streamlit_cloud.txt",
        "UDnet_dynamic.onnx"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    
    # Check git status
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  Uncommitted changes detected")
            print("   Please commit changes before deploying")
            return False
        print("✅ Git repository is clean")
    except FileNotFoundError:
        print("⚠️  Git not found - some deployment options may not work")
    
    return True

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud."""
    print("\n🌐 DEPLOYING TO STREAMLIT CLOUD")
    print("="*40)
    
    # Run the Streamlit Cloud deployment script
    try:
        result = subprocess.run([sys.executable, "deploy_streamlit_cloud.py"], 
                              check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to prepare Streamlit Cloud deployment")
        return False

def deploy_railway():
    """Deploy to Railway."""
    print("\n🚂 DEPLOYING TO RAILWAY")
    print("="*30)
    
    # Run the Railway deployment script
    try:
        result = subprocess.run([sys.executable, "deploy_railway.py"], 
                              check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to prepare Railway deployment")
        return False

def deploy_docker():
    """Deploy with Docker."""
    print("\n🐳 DEPLOYING WITH DOCKER")
    print("="*30)
    
    # Run the Docker deployment script
    try:
        result = subprocess.run([sys.executable, "deploy_docker.py"], 
                              check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to prepare Docker deployment")
        return False

def deploy_huggingface():
    """Deploy to Hugging Face Spaces."""
    print("\n🤗 DEPLOYING TO HUGGING FACE SPACES")
    print("="*40)
    
    print("📋 HUGGING FACE SPACES DEPLOYMENT STEPS:")
    print()
    print("1. 🌐 Go to: https://huggingface.co/new-space")
    print("2. 📝 Fill in the details:")
    print("   - Space name: underwater-image-enhancement")
    print("   - License: MIT")
    print("   - SDK: Streamlit")
    print("3. 📁 Upload these files:")
    print("   - streamlit_app.py")
    print("   - requirements_streamlit_cloud.txt")
    print("   - UDnet_dynamic.onnx")
    print("   - weights/UDnet.pth")
    print("4. 📄 Create README.md:")
    
    readme_content = """---
title: Underwater Image Enhancement
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.50.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# Underwater Image Enhancement

AI-powered underwater image enhancement using the UDnet model. Enhance underwater images with advanced deep learning techniques.

## Features

- 🖼️ Image enhancement
- 🎥 Video processing
- 📊 Quality metrics (PSNR, SSIM, UIQM)
- 🚀 Fast ONNX inference
- 📱 Mobile-friendly interface

## Usage

1. Upload an underwater image
2. Click "Enhance Image"
3. Download the enhanced result
4. View quality metrics

## Model

- **UDnet**: Variational autoencoder for underwater image enhancement
- **ONNX Runtime**: Optimized inference
- **Memory**: ~600MB peak usage
"""
    
    with open("README_HF.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_HF.md for Hugging Face Spaces")
    print("5. 🚀 Your space will be available at:")
    print("   https://huggingface.co/spaces/yourusername/underwater-image-enhancement")
    
    return True

def deploy_render():
    """Deploy to Render."""
    print("\n🎨 DEPLOYING TO RENDER")
    print("="*25)
    
    print("📋 RENDER DEPLOYMENT STEPS:")
    print()
    print("1. 🌐 Go to: https://render.com")
    print("2. 🔐 Sign up/Login with GitHub")
    print("3. 📁 Click 'New +' → 'Web Service'")
    print("4. 🔗 Connect your GitHub repository")
    print("5. ⚙️  Configure service:")
    print("   - Name: enhancement-pipeline")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r requirements_streamlit_cloud.txt")
    print("   - Start Command: streamlit run streamlit_app.py --server.headless true --server.port $PORT")
    print("6. 💾 Add environment variables:")
    print("   - PORT=8501")
    print("   - STREAMLIT_SERVER_HEADLESS=true")
    print("7. 🚀 Click 'Create Web Service'")
    print()
    print("⏱️  Deployment takes 5-10 minutes")
    print("💰 Cost: $7/month for starter plan")
    
    return True

def deploy_cloud_platforms():
    """Deploy to cloud platforms."""
    print("\n☁️  CLOUD PLATFORM DEPLOYMENT")
    print("="*35)
    
    print("🌐 AVAILABLE CLOUD PLATFORMS:")
    print()
    print("1. 🟠 AWS EC2")
    print("   - Launch t3.medium instance")
    print("   - Install Docker")
    print("   - Deploy with docker-compose")
    print()
    print("2. 🔵 Google Cloud Run")
    print("   - Build container image")
    print("   - Deploy to Cloud Run")
    print("   - Pay-per-use pricing")
    print()
    print("3. 🟣 Azure Container Instances")
    print("   - Build container image")
    print("   - Deploy to ACI")
    print("   - Enterprise features")
    print()
    print("📋 GENERAL STEPS:")
    print("1. 🐳 Build Docker image")
    print("2. 📤 Push to container registry")
    print("3. 🚀 Deploy to cloud platform")
    print("4. 🌐 Configure domain and SSL")
    print()
    print("💡 Use the Docker deployment script first!")
    
    return True

def deploy_jetson():
    """Deploy to Jetson devices."""
    print("\n🔧 JETSON EDGE DEPLOYMENT")
    print("="*30)
    
    print("📋 JETSON DEPLOYMENT STEPS:")
    print()
    print("1. 🔧 Prepare Jetson device:")
    print("   - Flash JetPack SDK")
    print("   - Install Python 3.8+")
    print("   - Install PyTorch for Jetson")
    print()
    print("2. 📦 Install dependencies:")
    print("   pip install -r jetson_requirements.txt")
    print()
    print("3. 🚀 Run application:")
    print("   python streamlit_app.py")
    print()
    print("4. 🌐 Access via network:")
    print("   http://jetson-ip:8501")
    print()
    print("📊 PERFORMANCE EXPECTATIONS:")
    print("   - Jetson Nano: ~2-3 FPS")
    print("   - Jetson Xavier NX: ~5-8 FPS")
    print("   - Jetson Orin NX: ~10-15 FPS")
    print()
    print("💡 Use TensorRT for best performance!")
    
    return True

def show_memory_requirements():
    """Show memory requirements for each platform."""
    print("\n💾 MEMORY REQUIREMENTS")
    print("="*25)
    print()
    print("📊 Model Memory Usage:")
    print("   - ONNX Model: 54.4 MB (total)")
    print("   - Peak Usage: 642.4 MB (512x512 images)")
    print("   - PyTorch Model: 1,303.8 MB (not recommended)")
    print()
    print("🌐 Platform Requirements:")
    print("   - Streamlit Cloud: 1GB (sufficient)")
    print("   - Hugging Face: 16GB (plenty)")
    print("   - Railway: 1GB (sufficient)")
    print("   - Render: 1GB (sufficient)")
    print("   - Docker: Depends on host")
    print("   - Jetson: 4-8GB (sufficient)")
    print()

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Enhancement Pipeline")
    parser.add_argument("--platform", choices=[
        "streamlit", "railway", "docker", "huggingface", 
        "render", "cloud", "jetson", "all"
    ], help="Deployment platform")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check prerequisites")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues and try again.")
        return False
    
    if args.check_only:
        print("\n✅ Prerequisites check complete!")
        return True
    
    # Show deployment options
    show_deployment_options()
    show_memory_requirements()
    
    # Handle platform selection
    if args.platform:
        platform = args.platform
    else:
        print("\n🎯 SELECT DEPLOYMENT PLATFORM:")
        print("1. Streamlit Cloud (Free, Easy)")
        print("2. Hugging Face Spaces (Free, ML-focused)")
        print("3. Railway (Production, $5/month)")
        print("4. Render (Production, $7/month)")
        print("5. Docker (Local/Cloud, Flexible)")
        print("6. Cloud Platforms (AWS/GCP/Azure)")
        print("7. Jetson Edge Deployment")
        print("8. Show all options")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        platform_map = {
            "1": "streamlit",
            "2": "huggingface", 
            "3": "railway",
            "4": "render",
            "5": "docker",
            "6": "cloud",
            "7": "jetson",
            "8": "all"
        }
        
        platform = platform_map.get(choice, "all")
    
    # Deploy based on platform
    success = True
    
    if platform == "streamlit" or platform == "all":
        success &= deploy_streamlit_cloud()
    
    if platform == "huggingface" or platform == "all":
        success &= deploy_huggingface()
    
    if platform == "railway" or platform == "all":
        success &= deploy_railway()
    
    if platform == "render" or platform == "all":
        success &= deploy_render()
    
    if platform == "docker" or platform == "all":
        success &= deploy_docker()
    
    if platform == "cloud" or platform == "all":
        success &= deploy_cloud_platforms()
    
    if platform == "jetson" or platform == "all":
        success &= deploy_jetson()
    
    if success:
        print("\n🎉 DEPLOYMENT PREPARATION COMPLETE!")
        print("="*40)
        print("✅ All selected platforms prepared successfully")
        print("🚀 Follow the instructions above to complete deployment")
    else:
        print("\n❌ Some deployment preparations failed")
        print("Please check the errors above and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
