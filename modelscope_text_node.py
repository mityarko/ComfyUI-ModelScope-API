import requests
import json
import time
import os
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: openai library not installed, text generation functionality will be unavailable")
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
            "default_text_model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "default_system_prompt": "You are a helpful assistant.",
            "default_user_prompt": "Hello",
            "api_token": ""
        }

def save_config(config):
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save configuration: {e}")
        return False

def load_api_token():
    try:
        cfg = load_config()
        return cfg.get("api_token", "").strip()
    except Exception as e:
        print(f"Failed to read token from config.json: {e}")
        return ""

def save_api_token(token):
    try:
        cfg = load_config()
        cfg["api_token"] = token
        return save_config(cfg)
    except Exception as e:
        print(f"Failed to save token: {e}")
        return False

class ModelScopeTextNode:
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
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_user_prompt", "Hello")
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "Please enter your ModelScope API Token",
                    "multiline": False
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_system_prompt", "You are a helpful assistant.")
                }),
                "model": (config.get("text_models", ["Qwen/Qwen3-Coder-480B-A35B-Instruct"]) + config.get("vision_models", []), {
                    "default": config.get("default_text_model", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 100,
                    "max": 8000
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "stream": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "ModelScopeAPI"

    def generate_text(self, user_prompt="", api_token="", system_prompt="You are a helpful assistant.", model="Qwen/Qwen3-Coder-480B-A35B-Instruct", max_tokens=2000, temperature=0.7, stream=True, seed=-1, error_message=""):
        if not OPENAI_AVAILABLE:
            return ("Please install openai library first: pip install openai",)
        
        if seed == -1:
            seed = np.random.randint(0, 2147483647)
        np.random.seed(seed % (2**32 - 1))
        
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
            print(f"💬 Starting text generation...")
            print(f"🤖 Model: {model}")
            print(f"📝 User Prompt: {user_prompt[:50]}...")
            print(f"⚙️ System Prompt: {system_prompt[:50]}...")
            print(f"🌡️ Temperature: {temperature}")
            print(f"📊 Max Tokens: {max_tokens}")
            print(f"⚡ Streaming: {stream}")
            print(f"🔢 Seed: {seed}")
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.ai/v1',
                api_key=api_token
            )
            
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ]
            
            print(f"🚀 Sending API request...")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            if stream:
                print("📡 Receiving streaming response...")
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end='', flush=True)
                
                print(f"\n✅ Streaming generation complete!")
                print(f"📄 Total length: {len(full_response)} characters")
                return (full_response,)
            else:
                result = response.choices[0].message.content
                print(f"✅ Text generation complete!")
                print(f"📄 Result length: {len(result)} characters")
                print(f"📝 Result preview: {result[:100]}...")
                return (result,)
            
        except Exception as e:
            error_msg = f"Text generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "ModelScopeTextNode": ModelScopeTextNode
    }
     
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeTextNode": "ModelScope Text Generation"
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
            return ("Please install openai library first to use text generation functionality: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "ModelScopeTextNode": OpenAINotInstalledNode
    }
 
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeTextNode": "ModelScope Text Generation (openai installation required)"
    }