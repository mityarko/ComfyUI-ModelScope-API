import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import tempfile
 
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: openai library not installed, Image-to-Text functionality will be unavailable")
    print("Please run: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None
 
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "default_model": "Qwen/Qwen-Image",
            "timeout": 720,
            "image_download_timeout": 30,
            "default_prompt": "A beautiful landscape",
            "api_token": ""
        }
 
def save_config(config):
    """Save configuration to modelscope_config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save configuration: {e}")
        return False
 
def load_api_token():
    """Read API Token only from modelscope_config.json"""
    try:
        cfg = load_config()
        return cfg.get("api_token", "").strip()
    except Exception as e:
        print(f"Failed to read token from config.json: {e}")
        return ""
 
def save_api_token(token):
    """Save API Token only to modelscope_config.json"""
    try:
        cfg = load_config()
        cfg["api_token"] = token.strip()
        return save_config(cfg)
    except Exception as e:
        print(f"Failed to save token: {e}")
        return False
 
def tensor_to_base64_url(image_tensor):
    try:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        if image_tensor.max() <= 1.0:
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"Image conversion failed: {e}")
        raise Exception(f"Image format conversion failed: {str(e)}")
 
class ModelScopeVisionNode:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        if not OPENAI_AVAILABLE:
            return {
                "required": {
                    "error_message": ("STRING", {
                        "default": "Please install openai library first: pip install openai",
                        "multiline": True
                    }),
                }
            }
        config = load_config()
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "Describe this image")
                }),
                "api_token": ("STRING", {
                    "default": "",
                    "placeholder": "Please enter your ModelScope API Token",
                    "multiline": False
                }),
            },
            "optional": {
                "model": (config.get("vision_models", ["stepfun-ai/step3"]), {
                    "default": config.get("default_vision_model", "stepfun-ai/step3")
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_image"
    CATEGORY = "ModelScopeAPI"
 
    def analyze_image(self, image=None, prompt="", api_token="", model="stepfun-ai/step3", max_tokens=1000, temperature=0.7, error_message=""):
        if not OPENAI_AVAILABLE:
            return ("Please install openai library first: pip install openai",)
        
        config = load_config()
        
        if not api_token or api_token.strip() == "":
            api_token = load_api_token()
            if not api_token or api_token.strip() == "":
                raise Exception("Please enter a valid API Token or ensure token is saved")
        
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("✅ API Token has been automatically saved to modelscope_config.json")
            else:
                print("⚠️ API Token saving failed, but it doesn't affect current usage")
        
        try:
            print(f"🔍 Starting image analysis...")
            print(f"📝 Prompt: {prompt}")
            print(f"🤖 Model: {model}")
            
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ Image converted to base64 format")
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.ai/v1',
                api_key=api_token
            )
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }]
            
            print(f"🚀 Sending API request...")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            description = response.choices[0].message.content
            print(f"✅ Analysis complete!")
            print(f"📄 Result: {description[:100]}...")
            
            return (description,)
            
        except Exception as e:
            error_msg = f"Image analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)
 
if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": ModelScopeVisionNode
    }
     
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope Vision Analysis"
    }
else:
    class OpenAINotInstalledNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "install_command": ("STRING", {
                        "default": "pip install openai",
                        "multiline": False
                    }),
                }
            }
        
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("message",)
        FUNCTION = "show_install_message"
        CATEGORY = "ModelScopeAPI"
        
        def show_install_message(self, install_command):
            return ("Please install openai library first to use Image-to-Text functionality: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": OpenAINotInstalledNode
    }
 
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope Vision Analysis (openai installation required)"
    }