#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("ModelScope API ComfyUI Plugin Dependency Installation Tool")
    print("=" * 60)
    
    # Check core dependencies
    core_deps = {
        'requests': 'requests',
        'PIL': 'pillow',
        'torch': 'torch',
        'numpy': 'numpy'
    }
    
    print("\n🔍 Checking core dependencies...")
    missing_core = []
    for import_name, package_name in core_deps.items():
        if check_package(import_name):
            print(f"✅ {package_name} is installed")
        else:
            print(f"❌ {package_name} is not installed")
            missing_core.append(package_name)
    
    # Check vision functionality dependencies
    print("\n🔍 Checking vision functionality dependencies...")
    vision_deps = {
        'openai': 'openai',
        'httpx': 'httpx[socks]',
        'socksio': 'socksio'
    }
    
    missing_vision = []
    for import_name, package_name in vision_deps.items():
        if check_package(import_name):
            print(f"✅ {package_name} is installed")
        else:
            print(f"❌ {package_name} is not installed")
            missing_vision.append(package_name)
    
    # Install missing dependencies
    all_missing = missing_core + missing_vision
    
    if not all_missing:
        print("\n🎉 All dependencies are already installed!")
        return
    
    print(f"\n📦 {len(all_missing)} dependency packages need to be installed:")
    for pkg in all_missing:
        print(f"  - {pkg}")
    
    response = input("\nInstall these dependencies now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting dependency installation...")
        success_count = 0
        
        for package in all_missing:
            print(f"\n📦 Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
                success_count += 1
            else:
                print(f"❌ {package} installation failed")
        
        print(f"\n📊 Installation results: {success_count}/{len(all_missing)} packages installed successfully")
        
        if success_count == len(all_missing):
            print("🎉 All dependencies installed! Please restart ComfyUI.")
        else:
            print("⚠️ Some dependencies failed to install. Please install manually or check your network connection.")
            print("\nManual installation commands:")
            for package in all_missing:
                print(f"  pip install {package}")
    else:
        print("\nInstallation cancelled.")
        print("\nManual installation commands:")
        for package in all_missing:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()