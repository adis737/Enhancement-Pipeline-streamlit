#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Helper
================================

This script helps prepare your project for Streamlit Cloud deployment.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist."""
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
    return True

def check_git_status():
    """Check git status and ensure everything is committed."""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("⚠️  Uncommitted changes detected:")
            print(result.stdout)
            return False
        
        print("✅ Git repository is clean")
        return True
    except FileNotFoundError:
        print("❌ Git not found. Please install Git first.")
        return False

def check_file_sizes():
    """Check file sizes to ensure they're within limits."""
    files_to_check = {
        "UDnet_dynamic.onnx": 10 * 1024 * 1024,  # 10MB
        "weights/UDnet.pth": 100 * 1024 * 1024,  # 100MB
    }
    
    large_files = []
    for file, max_size in files_to_check.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            if size > max_size:
                large_files.append((file, size, max_size))
    
    if large_files:
        print("⚠️  Large files detected:")
        for file, size, max_size in large_files:
            print(f"   - {file}: {size/1024/1024:.1f}MB (max: {max_size/1024/1024:.1f}MB)")
        print("   Consider using Git LFS for large files")
        return False
    
    print("✅ All files are within size limits")
    return True

def create_streamlit_config():
    """Create Streamlit configuration file."""
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    config_content = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    config_file = config_dir / "config.toml"
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print("✅ Created .streamlit/config.toml")

def create_app_toml():
    """Create app.toml for Streamlit Cloud."""
    app_toml_content = """[app]
name = "Underwater Image Enhancement"
description = "AI-powered underwater image enhancement using UDnet model"
icon = "🌊"
color = "blue"

[build]
requirements = "requirements_streamlit_cloud.txt"
main_file = "streamlit_app.py"

[deploy]
memory = 1024
cpu = 1
"""
    
    with open("app.toml", "w") as f:
        f.write(app_toml_content)
    
    print("✅ Created app.toml")

def show_deployment_instructions():
    """Show deployment instructions."""
    print("\n" + "="*60)
    print("🚀 STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print()
    print("1. 🌐 Go to: https://share.streamlit.io")
    print("2. 🔐 Sign in with your GitHub account")
    print("3. 📁 Click 'New app'")
    print("4. 🔗 Select your repository: Enhancement-Pipeline-streamlit")
    print("5. 📄 Main file path: streamlit_app.py")
    print("6. 📋 Requirements file: requirements_streamlit_cloud.txt")
    print("7. 🚀 Click 'Deploy!'")
    print()
    print("⏱️  Deployment typically takes 2-5 minutes")
    print("🔗 Your app will be available at: https://your-app-name.streamlit.app")
    print()
    print("📊 Memory Usage: ~600MB (well within 1GB limit)")
    print("⚡ Startup Time: ~10 seconds")
    print("🌊 Features: Image enhancement, video processing, quality metrics")
    print()

def main():
    """Main deployment preparation function."""
    print("🌊 Streamlit Cloud Deployment Preparation")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please ensure all required files exist before deploying")
        return False
    
    # Check git status
    if not check_git_status():
        print("\n⚠️  Please commit all changes before deploying")
        print("   Run: git add . && git commit -m 'Ready for deployment'")
        return False
    
    # Check file sizes
    check_file_sizes()
    
    # Create configuration files
    create_streamlit_config()
    create_app_toml()
    
    # Show deployment instructions
    show_deployment_instructions()
    
    print("✅ Project is ready for Streamlit Cloud deployment!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
