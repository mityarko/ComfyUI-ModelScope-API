#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json

def check_files():
    required_files = [
        '__init__.py',
        'modelscope_image_node.py',
        'modelscope_vision_node.py',
        'modelscope_text_node.py',
        'modelscope_image_caption_node.py',
        'modelscope_config.json',
        'README.md',
        'requirements.txt'
    ]
    
    print("📁 Checking file integrity...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (Missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dependencies():
    print("\n📦 Checking dependency packages...")
    
    deps = {
        'requests': 'Network requests',
        'PIL': 'Image processing',
        'torch': 'Deep learning framework',
        'numpy': 'Numerical computing',
        'openai': 'Text generation and Vision functionality',
        'httpx': 'Advanced HTTP client',
        'socksio': 'SOCKS proxy support'
    }
    
    missing_deps = []
    
    for dep, desc in deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep} ({desc})")
        except ImportError:
            print(f"❌ {dep} ({desc}) - Not installed")
            missing_deps.append(dep)
    
    return len(missing_deps) == 0, missing_deps

def check_proxy_support():
    print("\n🌐 Checking proxy support...")
    
    try:
        import httpx
        try:
            import socksio
            print("✅ SOCKS proxy support is installed")
            return True
        except ImportError:
            print("⚠️ SOCKS proxy support not installed, errors may occur if using a proxy")
            print("   Suggested: pip install httpx[socks] socksio")
            return False
    except ImportError:
        print("❌ httpx not installed")
        return False

def check_node_loading():
    print("\n🔧 Checking node loading...")

    # Mocking folder_paths to avoid errors during standalone check
    import sys
    from unittest.mock import MagicMock
    if 'folder_paths' not in sys.modules:
        sys.modules['folder_paths'] = MagicMock()
    
    try:
        from modelscope_image_node import ModelScopeImageNode
        node = ModelScopeImageNode()
        input_types = node.INPUT_TYPES()
        print("✅ Image Generation node loaded successfully")
        
        from modelscope_vision_node import ModelScopeVisionNode, OPENAI_AVAILABLE
        if OPENAI_AVAILABLE:
            vision_node = ModelScopeVisionNode()
            vision_input_types = vision_node.INPUT_TYPES()
            print("✅ Vision Analysis node loaded successfully")
        else:
            print("⚠️ Vision Analysis node loaded, but OpenAI library is unavailable")
        
        from modelscope_text_node import ModelScopeTextNode
        if OPENAI_AVAILABLE:
            text_node = ModelScopeTextNode()
            text_input_types = text_node.INPUT_TYPES()
            print("✅ Text Generation node loaded successfully")
        else:
            print("⚠️ Text Generation node loaded, but OpenAI library is unavailable")

        from modelscope_image_caption_node import ModelScopeImageCaptionNode
        image_caption_node = ModelScopeImageCaptionNode()
        image_caption_input_types = image_caption_node.INPUT_TYPES()
        print("✅ Image Captioning node loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Node loading failed: {e}")
        return False

def check_config():
    print("\n⚙️ Checking configuration file...")
    
    try:
        with open('modelscope_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = [
            'default_model',
            'timeout',
            'default_prompt'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ {key} (Missing)")
                missing_keys.append(key)
        
        return len(missing_keys) == 0
    except Exception as e:
        print(f"❌ Configuration file read failed: {e}")
        return False

def main():
    print("=" * 60)
    print("ModelScope API ComfyUI Plugin Installation Verification")
    print("=" * 60)
    
    checks = [
        ("File Integrity", check_files),
        ("Dependency Packages", lambda: check_dependencies()[0]),
        ("Proxy Support", check_proxy_support),
        ("Configuration File", check_config),
        ("Node Loading", check_node_loading),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name} check...")
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} check passed")
            else:
                print(f"❌ {check_name} check failed")
        except Exception as e:
            print(f"❌ {check_name} check errored: {e}")
        
        print("-" * 40)
    
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"\n📦 Missing dependency packages: {', '.join(missing_deps)}")
        print("Run the following command to install:")
        print("python install_dependencies.py")
        print("Or install manually:")
        for dep in missing_deps:
            if dep == 'httpx':
                print(f"  pip install httpx[socks]")
            else:
                print(f"  pip install {dep}")
    
    print(f"\n📊 Verification results: {passed}/{total} items passed")
    
    if passed >= total - 1:
        print("\n🎉 Plugin installation verification successful!")
        print("\n📋 Next steps:")
        print("1. Ensure the entire plugin folder is in ComfyUI/custom_nodes/ directory")
        print("2. Restart ComfyUI")
        print("3. Look for 'ModelScopeAPI' category in the node list")
        print("4. Have your ModelScope API Token ready")
        
        current_path = os.getcwd()
        if 'custom_nodes' in current_path:
            print("\n✅ Detected you are already in the ComfyUI custom_nodes directory")
            print("   Please restart ComfyUI to use the plugin")
        else:
            print(f"\n📁 Current path: {current_path}")
            print("   Please ensure you have copied the plugin to the correct ComfyUI directory")
            
        if not check_proxy_support():
            print("\n⚠️ Proxy support reminder:")
            print("   If you use a proxy to access the internet, suggested to install proxy support packages:")
            print("   pip install httpx[socks] socksio")
    else:
        print("\n⚠️ Plugin installation verification failed, please fix the issues above and retry")

if __name__ == "__main__":
    main()