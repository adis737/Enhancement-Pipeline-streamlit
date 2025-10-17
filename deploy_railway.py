#!/usr/bin/env python3
"""
Railway Deployment Helper
========================

This script helps deploy the Enhancement Pipeline to Railway.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def check_railway_cli():
    """Check if Railway CLI is installed."""
    try:
        result = subprocess.run(["railway", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Railway CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Railway CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Railway CLI not installed")
        return False

def install_railway_cli():
    """Install Railway CLI."""
    print("📦 Installing Railway CLI...")
    
    try:
        # Try npm first
        result = subprocess.run(["npm", "install", "-g", "@railway/cli"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Railway CLI installed via npm")
            return True
    except FileNotFoundError:
        print("⚠️  npm not found, trying alternative installation...")
    
    try:
        # Try curl installation
        result = subprocess.run([
            "curl", "-fsSL", "https://railway.app/install.sh"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Railway CLI installed via curl")
            return True
    except FileNotFoundError:
        print("❌ curl not found")
    
    print("❌ Failed to install Railway CLI")
    print("Please install manually from: https://docs.railway.app/develop/cli")
    return False

def create_railway_config():
    """Create Railway configuration files."""
    
    # Create railway.json
    railway_config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "streamlit run streamlit_app.py --server.headless true --server.port $PORT",
            "healthcheckPath": "/",
            "healthcheckTimeout": 100,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    print("✅ Created railway.json")
    
    # Create Procfile
    procfile_content = "web: streamlit run streamlit_app.py --server.headless true --server.port $PORT"
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    print("✅ Created Procfile")

def create_nixpacks_config():
    """Create Nixpacks configuration."""
    nixpacks_config = """[phases.setup]
nixPkgs = ['python39', 'pip']

[phases.install]
cmds = ['pip install -r requirements_streamlit_cloud.txt']

[phases.build]
cmds = ['echo "Build complete"']

[start]
cmd = 'streamlit run streamlit_app.py --server.headless true --server.port $PORT'
"""
    
    with open("nixpacks.toml", "w") as f:
        f.write(nixpacks_config)
    
    print("✅ Created nixpacks.toml")

def check_required_files():
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

def login_railway():
    """Login to Railway."""
    print("🔐 Logging into Railway...")
    
    try:
        result = subprocess.run(["railway", "login"], check=True)
        print("✅ Successfully logged into Railway")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to login to Railway: {e}")
        return False

def init_railway_project():
    """Initialize Railway project."""
    print("🚀 Initializing Railway project...")
    
    try:
        result = subprocess.run(["railway", "init"], check=True)
        print("✅ Railway project initialized")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize Railway project: {e}")
        return False

def deploy_to_railway():
    """Deploy to Railway."""
    print("🚀 Deploying to Railway...")
    
    try:
        result = subprocess.run(["railway", "up"], check=True)
        print("✅ Successfully deployed to Railway")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to deploy to Railway: {e}")
        return False

def get_railway_url():
    """Get Railway deployment URL."""
    try:
        result = subprocess.run(["railway", "domain"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            url = result.stdout.strip()
            print(f"🌐 Your app is available at: {url}")
            return url
    except Exception as e:
        print(f"⚠️  Could not get Railway URL: {e}")
    
    return None

def show_railway_commands():
    """Show useful Railway commands."""
    print("\n" + "="*60)
    print("🚂 RAILWAY COMMANDS")
    print("="*60)
    print()
    print("🔐 Login:")
    print("   railway login")
    print()
    print("🚀 Initialize project:")
    print("   railway init")
    print()
    print("📤 Deploy:")
    print("   railway up")
    print()
    print("🌐 Get domain:")
    print("   railway domain")
    print()
    print("📊 View logs:")
    print("   railway logs")
    print()
    print("🔧 Open dashboard:")
    print("   railway open")
    print()
    print("🛑 Stop service:")
    print("   railway down")
    print()
    print("💾 Set environment variables:")
    print("   railway variables set KEY=value")
    print()
    print("📋 List services:")
    print("   railway status")
    print()

def show_deployment_instructions():
    """Show deployment instructions."""
    print("\n" + "="*60)
    print("🚂 RAILWAY DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print()
    print("1. 🔐 Login to Railway:")
    print("   railway login")
    print()
    print("2. 🚀 Initialize project:")
    print("   railway init")
    print()
    print("3. 📤 Deploy:")
    print("   railway up")
    print()
    print("4. 🌐 Get your URL:")
    print("   railway domain")
    print()
    print("⏱️  Deployment typically takes 3-5 minutes")
    print("💰 Cost: $5/month for starter plan")
    print("💾 Memory: 1GB (sufficient for ONNX model)")
    print("🌊 Features: Image enhancement, video processing, quality metrics")
    print()

def main():
    """Main Railway deployment function."""
    print("🚂 Railway Deployment Preparation")
    print("="*40)
    
    # Check required files
    if not check_required_files():
        return False
    
    # Check Railway CLI
    if not check_railway_cli():
        print("Installing Railway CLI...")
        if not install_railway_cli():
            return False
    
    # Create configuration files
    create_railway_config()
    create_nixpacks_config()
    
    # Show deployment instructions
    show_deployment_instructions()
    show_railway_commands()
    
    # Ask user if they want to deploy now
    response = input("\n🚀 Do you want to deploy to Railway now? (y/n): ").lower()
    if response == 'y':
        if login_railway():
            if init_railway_project():
                if deploy_to_railway():
                    get_railway_url()
    
    print("✅ Railway deployment preparation complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
