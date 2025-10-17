#!/usr/bin/env python3
"""
Enhancement Pipeline - Simple Deployment Script
==============================================

This script provides a simple interface for deploying the Enhancement Pipeline
to various platforms without Unicode issues.
"""

import os
import sys
import subprocess
import argparse

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
    print("4. Docker (Local/Cloud, Flexible)")
    print("   - Run anywhere")
    print("   - Full control")
    print("   - Deploy to any cloud")
    print("   - Perfect for development")
    print()

def check_prerequisites():
    """Check deployment prerequisites."""
    print("CHECKING PREREQUISITES...")
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
        print("ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("SUCCESS: All required files found")
    
    # Check git status
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("WARNING: Uncommitted changes detected")
            print("   Please commit changes before deploying")
            return False
        print("SUCCESS: Git repository is clean")
    except FileNotFoundError:
        print("WARNING: Git not found - some deployment options may not work")
    
    return True

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud."""
    print("\nDEPLOYING TO STREAMLIT CLOUD")
    print("="*40)
    
    print("STREAMLIT CLOUD DEPLOYMENT STEPS:")
    print()
    print("1. Go to: https://share.streamlit.io")
    print("2. Sign in with your GitHub account")
    print("3. Click 'New app'")
    print("4. Select your repository: Enhancement-Pipeline-streamlit")
    print("5. Main file path: streamlit_app.py")
    print("6. Requirements file: requirements_streamlit_cloud.txt")
    print("7. Click 'Deploy!'")
    print()
    print("Deployment typically takes 2-5 minutes")
    print("Your app will be available at: https://your-app-name.streamlit.app")
    print()
    print("Memory Usage: ~600MB (well within 1GB limit)")
    print("Startup Time: ~10 seconds")
    print("Features: Image enhancement, video processing, quality metrics")
    print()
    
    return True

def deploy_railway():
    """Deploy to Railway."""
    print("\nDEPLOYING TO RAILWAY")
    print("="*30)
    
    print("RAILWAY DEPLOYMENT STEPS:")
    print()
    print("1. Install Railway CLI:")
    print("   npm install -g @railway/cli")
    print()
    print("2. Login to Railway:")
    print("   railway login")
    print()
    print("3. Initialize project:")
    print("   railway init")
    print()
    print("4. Deploy:")
    print("   railway up")
    print()
    print("5. Get your URL:")
    print("   railway domain")
    print()
    print("Deployment typically takes 3-5 minutes")
    print("Cost: $5/month for starter plan")
    print("Memory: 1GB (sufficient for ONNX model)")
    print()
    
    return True

def deploy_docker():
    """Deploy with Docker."""
    print("\nDEPLOYING WITH DOCKER")
    print("="*30)
    
    print("DOCKER DEPLOYMENT STEPS:")
    print()
    print("1. Build Docker image:")
    print("   docker build -t enhancement-pipeline .")
    print()
    print("2. Run container:")
    print("   docker run -p 8501:8501 enhancement-pipeline")
    print()
    print("3. Access application:")
    print("   http://localhost:8501")
    print()
    print("For production deployment:")
    print("1. Push to container registry")
    print("2. Deploy to cloud platform")
    print("3. Configure domain and SSL")
    print()
    
    return True

def deploy_huggingface():
    """Deploy to Hugging Face Spaces."""
    print("\nDEPLOYING TO HUGGING FACE SPACES")
    print("="*40)
    
    print("HUGGING FACE SPACES DEPLOYMENT STEPS:")
    print()
    print("1. Go to: https://huggingface.co/new-space")
    print("2. Fill in the details:")
    print("   - Space name: underwater-image-enhancement")
    print("   - License: MIT")
    print("   - SDK: Streamlit")
    print("3. Upload these files:")
    print("   - streamlit_app.py")
    print("   - requirements_streamlit_cloud.txt")
    print("   - UDnet_dynamic.onnx")
    print("   - weights/UDnet.pth")
    print("4. Your space will be available at:")
    print("   https://huggingface.co/spaces/yourusername/underwater-image-enhancement")
    print()
    
    return True

def show_memory_requirements():
    """Show memory requirements for each platform."""
    print("\nMEMORY REQUIREMENTS")
    print("="*25)
    print()
    print("Model Memory Usage:")
    print("   - ONNX Model: 54.4 MB (total)")
    print("   - Peak Usage: 642.4 MB (512x512 images)")
    print("   - PyTorch Model: 1,303.8 MB (not recommended)")
    print()
    print("Platform Requirements:")
    print("   - Streamlit Cloud: 1GB (sufficient)")
    print("   - Hugging Face: 16GB (plenty)")
    print("   - Railway: 1GB (sufficient)")
    print("   - Docker: Depends on host")
    print()

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Enhancement Pipeline")
    parser.add_argument("--platform", choices=[
        "streamlit", "railway", "docker", "huggingface", "all"
    ], help="Deployment platform")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check prerequisites")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites not met. Please fix issues and try again.")
        return False
    
    if args.check_only:
        print("\nSUCCESS: Prerequisites check complete!")
        return True
    
    # Show deployment options
    show_deployment_options()
    show_memory_requirements()
    
    # Handle platform selection
    if args.platform:
        platform = args.platform
    else:
        print("\nSELECT DEPLOYMENT PLATFORM:")
        print("1. Streamlit Cloud (Free, Easy)")
        print("2. Hugging Face Spaces (Free, ML-focused)")
        print("3. Railway (Production, $5/month)")
        print("4. Docker (Local/Cloud, Flexible)")
        print("5. Show all options")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        platform_map = {
            "1": "streamlit",
            "2": "huggingface", 
            "3": "railway",
            "4": "docker",
            "5": "all"
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
    
    if platform == "docker" or platform == "all":
        success &= deploy_docker()
    
    if success:
        print("\nSUCCESS: DEPLOYMENT PREPARATION COMPLETE!")
        print("="*40)
        print("All selected platforms prepared successfully")
        print("Follow the instructions above to complete deployment")
    else:
        print("\nERROR: Some deployment preparations failed")
        print("Please check the errors above and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
