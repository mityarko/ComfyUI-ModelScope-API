import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import re
from .modelscope_image_node import load_config, save_config, tensor_to_base64_url

# Check if openai library is available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# API Token management functions interacting only with modelscope_config.json
def load_api_tokens():
    try:
        cfg = load_config()
        tokens_from_cfg = cfg.get("api_tokens", [])
        if tokens_from_cfg and isinstance(tokens_from_cfg, list):
            return [token.strip() for token in tokens_from_cfg if token.strip()]
    except Exception as e:
        print(f"Failed to read tokens from config: {e}")
    return []

def save_api_tokens(tokens):
    try:
        cfg = load_config()
        cfg["api_tokens"] = tokens
        return save_config(cfg)
    except Exception as e:
        print(f"Failed to save tokens to config: {e}")
        return False

class ModelScopeImageCaptionNode:
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
        saved_tokens = load_api_tokens()
        # Define supported models list
        supported_models = [
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ]
        return {
            "required": {
                "api_tokens": ("STRING", {
                    "default": f"***Saved {len(saved_tokens)} tokens***" if saved_tokens else "",
                    "placeholder": "Please enter API Token (supports multiple, separated by comma/newline)",
                    "multiline": True
                }),
            },
            "optional": {
                # Image set as optional input
                "image": ("IMAGE", {"optional": True}),
                "prompt1": ("STRING", {
                    "multiline": True,
                    "default": "Describe the content of this image in detail, including the subject, background, colors, style, and other information."
                }),
                "prompt2": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model": (supported_models, {
                    "default": "Qwen/Qwen3-VL-8B-Instruct"
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
                # New seed option (consistent with image generation nodes: default -1 means random)
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "generate_caption"
    CATEGORY = "ModelScopeAPI"

    def parse_api_tokens(self, token_input):
        """Parse multiple input API Tokens (supports comma, semicolon, newline separators)"""
        if not token_input or token_input.strip() == "" or token_input.strip().startswith("***Saved"):
            return load_api_tokens()
        
        # Support multiple separators for splitting Tokens
        tokens = re.split(r'[,;\n]+', token_input)
        return [token.strip() for token in tokens if token.strip()]
    
    def create_blank_image(self, width=64, height=64):
        """Create a blank image tensor (conforms to ComfyUI image format requirements)"""
        # Create a white background RGB image
        blank_np = np.ones((height, width, 3), dtype=np.uint8) * 255
        # Convert to ComfyUI format tensor (batch, height, width, channels)
        blank_tensor = torch.from_numpy(blank_np).unsqueeze(0).float() / 255.0
        return blank_tensor

    def generate_caption(self, image=None, api_tokens="", prompt1="Describe this image in detail", prompt2="", model="Qwen/Qwen3-VL-8B-Instruct", max_tokens=1000, temperature=0.7, seed=-1):
        if not OPENAI_AVAILABLE:
            return ("Please install openai library first: pip install openai",)
        
        # Apply seed (-1 uses random seed)
        if seed == -1:
            seed = np.random.randint(0, 2147483647)
        np.random.seed(seed % (2**32 - 1))
        
        # Handle cases where input image is empty
        if image is None:
            print("⚠️ No input image, automatically generating blank image as input")
            image = self.create_blank_image()
        
        # Handle prompt merging
        prompt_parts = []
        if prompt1.strip():
            prompt_parts.append(prompt1.strip())
        if prompt2.strip():
            prompt_parts.append(prompt2.strip())
        
        if not prompt_parts:
            prompt = "Describe the content of this image in detail, including the subject, background, colors, style, and other information."
        else:
            prompt = ", ".join(prompt_parts)
        
        # Parse Token list
        tokens = self.parse_api_tokens(api_tokens)
        if not tokens:
            raise Exception("Please provide at least one valid API Token")
        
        # Save new Token (if changed)
        if api_tokens.strip() != "" and not api_tokens.strip().startswith("***Saved"):
            if save_api_tokens(tokens):
                print(f"✅ Saved {len(tokens)} API Tokens")
            else:
                print("⚠️ API Token saving failed, but it doesn't affect current usage")
        
        try:
            print(f"🔍 Starting image description generation...")
            print(f"📝 Prompt: {prompt}")
            print(f"🤖 Model: {model}")
            print(f"🔑 Available Token count: {len(tokens)}")
            print(f"🌱 Seed: {seed}")
            
            # Convert image to base64 format
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ Image converted to base64 format")
            
            # Build message body
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
            
            # Iterate through each Token
            last_exception = None
            for i, token in enumerate(tokens):
                try:
                    print(f"🔄 Attempting to use token {i+1}/{len(tokens)}...")
                    
                    client = OpenAI(
                        base_url='https://api-inference.modelscope.ai/v1',
                        api_key=token
                    )
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                    
                    description = response.choices[0].message.content
                    print(f"✅ Token {i+1} call successful!")
                    print(f"📄 Result preview: {description[:100]}...")
                    return (description,)
                    
                except Exception as e:
                    last_exception = e
                    print(f"❌ Token {i+1} call failed: {str(e)}")
                    if i < len(tokens) - 1:
                        print(f"⏳ Preparing to try next token...")
            
            # All Tokens failed
            raise Exception(f"All tokens failed: {str(last_exception)}")
            
        except Exception as e:
            error_msg = f"Image description generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "ModelScopeImageCaptionNode": ModelScopeImageCaptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageCaptionNode": "ModelScope Image Captioning"
}