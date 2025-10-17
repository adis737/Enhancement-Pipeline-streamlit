#!/usr/bin/env python3
"""
Docker Deployment Helper
=======================

This script helps build and deploy the Enhancement Pipeline using Docker.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed. Please install Docker first.")
        return False

def check_docker_running():
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(["docker", "info"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker daemon is running")
            return True
        else:
            print("❌ Docker daemon is not running")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker daemon: {e}")
        return False

def create_dockerfile():
    """Create optimized Dockerfile."""
    dockerfile_content = """# Multi-stage build for Enhancement Pipeline
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_streamlit_cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements_streamlit_cloud.txt

# Copy application files
COPY streamlit_app.py .
COPY udnet_infer.py .
COPY model_utils/ ./model_utils/
COPY UDnet_dynamic.onnx .
COPY weights/ ./weights/

# Create directories for outputs
RUN mkdir -p static/outputs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile")

def create_dockerignore():
    """Create .dockerignore file."""
    dockerignore_content = """# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode
.idea
*.swp
*.swo
*~

# Project specific
*.md
*.txt
!requirements_streamlit_cloud.txt
demo_*.py
test_*.py
*_monitor.py
deploy_*.py
check_*.py
build_*.py
export_*.py
run_*.py
jetson_*.py
*.jpg
*.png
*.mp4
static/outputs/*
!static/outputs/.gitkeep
"""
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    
    print("✅ Created .dockerignore")

def create_docker_compose():
    """Create docker-compose.yml file."""
    compose_content = """version: '3.8'

services:
  enhancement-pipeline:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    volumes:
      - ./static/outputs:/app/static/outputs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add a reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - enhancement-pipeline
    restart: unless-stopped
    profiles:
      - production
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ Created docker-compose.yml")

def build_docker_image():
    """Build Docker image."""
    print("🔨 Building Docker image...")
    
    try:
        result = subprocess.run([
            "docker", "build", 
            "-t", "enhancement-pipeline:latest",
            "-t", "enhancement-pipeline:$(date +%Y%m%d)",
            "."
        ], check=True)
        
        print("✅ Docker image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build Docker image: {e}")
        return False

def run_docker_container():
    """Run Docker container."""
    print("🚀 Starting Docker container...")
    
    try:
        result = subprocess.run([
            "docker", "run", 
            "-d",
            "--name", "enhancement-pipeline",
            "-p", "8501:8501",
            "--restart", "unless-stopped",
            "enhancement-pipeline:latest"
        ], check=True)
        
        print("✅ Docker container started successfully")
        print("🌐 Application available at: http://localhost:8501")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Docker container: {e}")
        return False

def show_docker_commands():
    """Show useful Docker commands."""
    print("\n" + "="*60)
    print("🐳 DOCKER COMMANDS")
    print("="*60)
    print()
    print("📦 Build image:")
    print("   docker build -t enhancement-pipeline .")
    print()
    print("🚀 Run container:")
    print("   docker run -p 8501:8501 enhancement-pipeline")
    print()
    print("🔄 Run with docker-compose:")
    print("   docker-compose up -d")
    print()
    print("📊 View logs:")
    print("   docker logs enhancement-pipeline")
    print()
    print("🛑 Stop container:")
    print("   docker stop enhancement-pipeline")
    print()
    print("🗑️  Remove container:")
    print("   docker rm enhancement-pipeline")
    print()
    print("🔍 Check running containers:")
    print("   docker ps")
    print()
    print("💾 Save image:")
    print("   docker save enhancement-pipeline > enhancement-pipeline.tar")
    print()
    print("📥 Load image:")
    print("   docker load < enhancement-pipeline.tar")
    print()

def main():
    """Main Docker deployment function."""
    print("🐳 Docker Deployment Preparation")
    print("="*40)
    
    # Check Docker
    if not check_docker():
        return False
    
    if not check_docker_running():
        print("Please start Docker daemon and try again")
        return False
    
    # Create Docker files
    create_dockerfile()
    create_dockerignore()
    create_docker_compose()
    
    # Build image
    if not build_docker_image():
        return False
    
    # Ask user if they want to run the container
    response = input("\n🚀 Do you want to start the container now? (y/n): ").lower()
    if response == 'y':
        run_docker_container()
    
    # Show commands
    show_docker_commands()
    
    print("✅ Docker deployment preparation complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
