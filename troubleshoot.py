#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelScope API ComfyUI Plugin Troubleshooting Tool
Automatically diagnose and resolve common issues
"""

import os
import sys
import subprocess
import json

def print_header(title):
    """Print header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """Print section title"""
    print(f"\n🔍 {title}")
    print("-" * 40)

def run_command(command, description):
    """Run command and return results"""
    print(f"📋 {description}")
    print(f"💻 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout.strip():
                print(f"📄 Output: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print("❌ Failed")
            if result.stderr.strip():
                print(f"🚨 Error: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        return False, "Command execution timed out"
    except Exception as e:
        print(f"💥 Exception: {str(e)}")
        return False, str(e)

def check_python_environment():
    """Check Python environment"""
    print_section("Python Environment Check")
    
    # Python version
    run_command("python --version", "Check Python version")
    
    # pip version
    run_command("pip --version", "Check pip version")
    
    # Installed packages
    print("\n📦 Checking key package installation status:")
    packages = ['requests', 'PIL', 'torch', 'numpy', 'openai', 'httpx', 'socksio']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (Not installed)")

def check_files():
    """Check file integrity"""
    print_section("File Integrity Check")
    
    required_files = [
        '__init__.py',
        'modelscope_image_node.py',
        'modelscope_vision_node.py',
        'modelscope_text_node.py',
        'modelscope_image_caption_node.py',
        'modelscope_config.json',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} (Missing)")

def check_config():
    """Check configuration file"""
    print_section("Configuration File Check")
    
    try:
        with open('modelscope_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✅ modelscope_config.json format is correct")
        
        # Check key configuration items
        key_configs = [
            'default_model',
            'timeout',
            'default_prompt'
        ]
        
        for key in key_configs:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ {key} (Missing)")
                
    except Exception as e:
        print(f"❌ modelscope_config.json read failed: {e}")

def check_network():
    """Check network connection"""
    print_section("Network Connection Check")
    
    # Check basic network connection
    # Note: ping command might differ between OS, but this is a common one
    run_command("ping -c 3 8.8.8.8" if os.name != 'nt' else "ping -n 3 8.8.8.8", "Check basic network connection")
    
    # Check API server connection
    try:
        import requests
        response = requests.get('https://api-inference.modelscope.cn', timeout=10)
        print(f"✅ API server connection normal (Status code: {response.status_code})")
    except Exception as e:
        print(f"❌ API server connection failed: {e}")
    
    # Check proxy settings
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY']
    print("\n🌐 Proxy environment variables:")
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚪ {var}: Not set")

def check_token():
    """Check API Token"""
    print_section("API Token Check")
    
    token_sources = ['modelscope_config.json']
    token_found = False
    
    for source in token_sources:
        if source == 'modelscope_config.json':
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Check api_token (singular) and api_tokens (plural)
                token = config.get('api_token', '').strip()
                if token:
                    print(f"✅ Found token in {source} (Length: {len(token)})")
                    token_found = True

                tokens = config.get('api_tokens', [])
                if tokens and any(t.strip() for t in tokens):
                    print(f"✅ Found {len(tokens)} tokens in {source}")
                    token_found = True

                if not token_found:
                    print(f"⚪ No tokens found in {source}")
            except Exception as e:
                print(f"❌ Read {source} failed: {e}")
    
    if not token_found:
        print("❌ No valid API token found")

def run_diagnostic_tests():
    """Run diagnostic tests"""
    print_section("Diagnostic Tests")
    
    tests = [
        ("python verify_installation.py", "Run installation verification"),
    ]
    
    for command, description in tests:
        script_file = command.split()[1]
        if os.path.exists(script_file):
            success, output = run_command(command, description)
            if not success:
                print(f"⚠️ {description} failed, please check detailed output")
        else:
            print(f"⚪ {script_file} does not exist, skipping test")

def suggest_solutions():
    """Suggest solutions"""
    print_section("Suggested Solutions")
    
    solutions = [
        "🔧 Install missing dependencies: python install_dependencies.py",
        "🔍 Verify installation: python verify_installation.py",
        "📖 View README for more information",
        "🔄 Restart ComfyUI to load updates",
        "🧹 Clean Python cache: rm -rf __pycache__ (Linux/Mac) or del /s /q __pycache__ (Windows)",
    ]
    
    for solution in solutions:
        print(solution)

def main():
    print_header("ModelScope API ComfyUI Plugin Troubleshooting Tool")
    
    print("🚀 Starting comprehensive diagnosis...")
    
    # Run all checks
    check_python_environment()
    check_files()
    check_config()
    check_network()
    check_token()
    run_diagnostic_tests()
    suggest_solutions()
    
    print_header("Diagnosis Complete")
    
    print("\n💡 Based on the diagnosis results above:")
    print("1. If missing dependencies were found, run: python install_dependencies.py")
    print("2. If there are network issues, check your proxy settings")
    print("3. If token issues were found, re-enter your API token in the nodes")
    print("4. If files are missing, re-download the plugin")
    print("5. After fixing issues, restart ComfyUI")
    
    print("\n📞 If issues persist:")
    print("- Check the ComfyUI console for complete error logs")
    print("- Try testing in a different network environment")
    print("- Confirm ComfyUI version compatibility")

if __name__ == "__main__":
    main()