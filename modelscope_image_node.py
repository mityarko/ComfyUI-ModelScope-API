import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import folder_paths
import base64
import tempfile
import re

# -------------------------- Core Configuration Management --------------------------
def load_config():
    """Load configuration from modelscope_config.json, ensuring prioritized use of lora_presets from the config file."""
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    default_config = {
        "default_model": "Qwen/Qwen-Image",
        "timeout": 720,
        "image_download_timeout": 30,
        "default_prompt": "A beautiful landscape",
        "default_negative_prompt": "",
        "default_width": 512,
        "default_height": 512,
        "default_seed": -1,
        "default_steps": 30,
        "default_guidance": 7.5,
        "default_lora_weight": 0.8,
        "image_models": ["Qwen/Qwen-Image"],
        "image_edit_models": ["Qwen/Qwen-Image-Edit"],
        "lora_presets": [
            {"name": "No LoRA", "model_id": "", "weight": 0.8}
        ],
        "api_tokens": []
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Ensure all necessary fields exist in the config file, supplementing with default values if missing
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    except Exception as e:
        print(f"Failed to read config file, using default configuration: {e}")
        return default_config

def save_config(config: dict) -> bool:
    """Save configuration to modelscope_config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save config file: {e}")
        return False

# -------------------------- API Token Management --------------------------
def save_api_tokens(tokens):
    try:
        cfg = load_config()
        cfg["api_tokens"] = tokens
        return save_config(cfg)
    except Exception as e:
        print(f"Failed to save API tokens: {e}")
        return False

def load_api_tokens():
    try:
        cfg = load_config()
        tokens_from_cfg = cfg.get("api_tokens", [])
        if tokens_from_cfg and isinstance(tokens_from_cfg, list):
            return [token.strip() for token in tokens_from_cfg if token.strip()]
        return []
    except Exception as e:
        print(f"Failed to load API tokens: {e}")
        return []

def parse_api_tokens(token_input):
    if not token_input or token_input.strip() == "" or token_input.strip().startswith("***Saved"):
        return load_api_tokens()
    
    tokens = re.split(r'[,;\n]+', token_input)
    return [token.strip() for token in tokens if token.strip()]

# -------------------------- Image Conversion Tools --------------------------
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
        raise Exception(f"Image format conversion failed: {str(e)}")

# -------------------------- LoRA Preset Management Node --------------------------
class ModelScopeLoraPresetNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Load LoRA preset list from config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        preset_names = [preset.get("name", "No LoRA") for preset in lora_presets]
        
        return {
            "required": {
                "action": (["View Presets", "Add Preset", "Delete Preset", "Save Preset"], {"default": "View Presets"}),
            },
            "optional": {
                "preset_name": ("STRING", {"default": "Custom LoRA", "label": "Preset Name"}),
                "lora_model_id": ("STRING", {"default": "", "label": "LoRA Model ID", "placeholder": "e.g., qiyuanai/TikTok_Xiaohongshu_career_line_beauty_v1"}),
                "default_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Default Weight"}),
                "target_preset": (preset_names, {"default": preset_names[0] if preset_names else "No LoRA", "label": "Target Preset"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_model_id", "lora_weight", "preset_info")
    FUNCTION = "manage_lora_presets"
    CATEGORY = "ModelScopeAPI/LoRA"
    
    def manage_lora_presets(self, action, preset_name="", lora_model_id="", default_weight=0.8, target_preset=""):
        # All operations are based on LoRA presets in the config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        preset_info = f"Total {len(lora_presets)} LoRA presets currently"
        
        if action == "View Presets":
            info_lines = ["=== LoRA Preset List ==="]
            for i, preset in enumerate(lora_presets):
                info_lines.append(f"{i+1}. {preset.get('name')} | ID: {preset.get('model_id')} | Weight: {preset.get('weight')}")
            preset_info = "\n".join(info_lines)
            selected_preset = next((p for p in lora_presets if p.get("name") == target_preset), {"model_id": "", "weight": 0.8})
            return (selected_preset.get("model_id"), selected_preset.get("weight"), preset_info)
        
        elif action == "Add Preset":
            if not preset_name or preset_name.strip() == "":
                raise Exception("Preset name cannot be empty")
            
            if any(p.get("name") == preset_name for p in lora_presets):
                raise Exception(f"Preset named {preset_name} already exists")
            
            new_preset = {
                "name": preset_name.strip(),
                "model_id": lora_model_id.strip(),
                "weight": float(default_weight)
            }
            lora_presets.append(new_preset)
            config["lora_presets"] = lora_presets
            save_config(config)
            preset_info = f"Successfully added preset: {preset_name} | ID: {lora_model_id}"
            return (lora_model_id, default_weight, preset_info)
        
        elif action == "Delete Preset":
            if target_preset == "No LoRA":
                raise Exception("Cannot delete the default 'No LoRA' preset")
            
            original_count = len(lora_presets)
            lora_presets = [p for p in lora_presets if p.get("name") != target_preset]
            if len(lora_presets) == original_count:
                raise Exception(f"Preset not found: {target_preset}")
            
            config["lora_presets"] = lora_presets
            save_config(config)
            preset_info = f"Successfully deleted preset: {target_preset}"
            return ("", 0.8, preset_info)
        
        elif action == "Save Preset":
            updated = False
            for i, preset in enumerate(lora_presets):
                if preset.get("name") == target_preset:
                    lora_presets[i]["model_id"] = lora_model_id.strip()
                    lora_presets[i]["weight"] = float(default_weight)
                    updated = True
                    break
            
            if not updated:
                raise Exception(f"Preset not found: {target_preset}")
            
            config["lora_presets"] = lora_presets
            save_config(config)
            preset_info = f"Successfully updated preset: {target_preset} | New ID: {lora_model_id} | New Weight: {default_weight}"
            return (lora_model_id, default_weight, preset_info)
        
        return ("", 0.8, preset_info)

# -------------------------- Single LoRA Loader Node --------------------------
class ModelScopeSingleLoraLoaderNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Load LoRA preset options from config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        preset_options = [preset.get("name", "No LoRA") for preset in lora_presets]
        
        return {
            "required": {
                "lora_preset": (preset_options, {"default": preset_options[0], "label": "LoRA Preset"}),
            },
            "optional": {
                "lora_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Custom Weight"}),
                "use_custom_weight": ("BOOLEAN", {"default": False, "label_on": "Use Custom Weight", "label_off": "Use Preset Weight"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("lora_id", "lora_weight")
    FUNCTION = "load_single_lora"
    CATEGORY = "ModelScopeAPI/LoRA"
    
    def load_single_lora(self, lora_preset, lora_weight=0.8, use_custom_weight=False):
        # Read selected LoRA info from config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        
        selected_preset = next((p for p in lora_presets if p.get("name") == lora_preset), {"model_id": "", "weight": 0.8})
        lora_id = selected_preset.get("model_id", "")
        final_weight = lora_weight if use_custom_weight else selected_preset.get("weight", 0.8)
        
        return (lora_id, final_weight)

# -------------------------- Multi LoRA Loader Node --------------------------
class ModelScopeMultiLoraLoaderNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # Load LoRA preset options from config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        preset_options = [preset.get("name", "No LoRA") for preset in lora_presets]
        
        return {
            "required": {
                "lora1_preset": (preset_options, {"default": preset_options[0], "label": "LoRA 1 Preset"}),
                "lora2_preset": (preset_options, {"default": preset_options[0], "label": "LoRA 2 Preset"}),
                "lora3_preset": (preset_options, {"default": preset_options[0], "label": "LoRA 3 Preset"}),
            },
            "optional": {
                "lora1_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 1 Weight"}),
                "lora2_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 2 Weight"}),
                "lora3_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 3 Weight"}),
                "lora1_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA1 Custom Weight", "label_off": "Preset Weight"}),
                "lora2_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA2 Custom Weight", "label_off": "Preset Weight"}),
                "lora3_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA3 Custom Weight", "label_off": "Preset Weight"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("lora1_id", "lora2_id", "lora3_id", "lora1_w", "lora2_w", "lora3_w")
    FUNCTION = "load_multi_lora"
    CATEGORY = "ModelScopeAPI/LoRA"
    
    def load_multi_lora(self, lora1_preset, lora2_preset, lora3_preset,
                        lora1_weight=0.8, lora2_weight=0.8, lora3_weight=0.8,
                        lora1_use_custom=False, lora2_use_custom=False, lora3_use_custom=False):
        # Read multiple LoRA info from config file
        config = load_config()
        lora_presets = config.get("lora_presets", [])
        
        def get_lora_info(preset_name, custom_weight, use_custom):
            preset = next((p for p in lora_presets if p.get("name") == preset_name), {"model_id": "", "weight": 0.8})
            model_id = preset.get("model_id", "")
            final_weight = custom_weight if use_custom else preset.get("weight", 0.8)
            return model_id, final_weight
        
        lora1_id, lora1_w = get_lora_info(lora1_preset, lora1_weight, lora1_use_custom)
        lora2_id, lora2_w = get_lora_info(lora2_preset, lora2_weight, lora2_use_custom)
        lora3_id, lora3_w = get_lora_info(lora3_preset, lora3_weight, lora3_use_custom)
        
        return (lora1_id, lora2_id, lora3_id, lora1_w, lora2_w, lora3_w)

# -------------------------- Image Generation Node --------------------------
class ModelScopeImageNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "A beautiful landscape")
                }),
                "api_tokens": ("STRING", {
                    "default": "***Saved {} tokens***".format(len(saved_tokens)) if saved_tokens else "",
                    "placeholder": "Please enter API Token (supports multiple, separated by comma/newline)" if not saved_tokens else "Leave blank to use saved tokens",
                    "multiline": True
                }),
            },
            "optional": {
                "model": (config.get("image_models", ["Qwen/Qwen-Image"]), {
                    "default": config.get("default_model", "Qwen/Qwen-Image")
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_negative_prompt", "")
                }),
                "width": ("INT", {
                    "default": config.get("default_width", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": config.get("default_height", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "seed": ("INT", {
                    "default": config.get("default_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
                "steps": ("INT", {
                    "default": config.get("default_steps", 30),
                    "min": 1,
                    "max": 100
                }),
                "guidance": ("FLOAT", {
                    "default": config.get("default_guidance", 7.5),
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
                "lora1_id": ("STRING", {"default": "", "label": "LoRA1 Model ID"}),
                "lora1_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA1 Weight"}),
                "lora2_id": ("STRING", {"default": "", "label": "LoRA2 Model ID"}),
                "lora2_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA2 Weight"}),
                "lora3_id": ("STRING", {"default": "", "label": "LoRA3 Model ID"}),
                "lora3_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA3 Weight"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScopeAPI"
    
    def generate_image(self, prompt, api_tokens, model="Qwen/Qwen-Image", negative_prompt="", width=512, height=512, seed=-1, steps=30, guidance=7.5,
                       lora1_id="", lora1_w=0.8, lora2_id="", lora2_w=0.8, lora3_id="", lora3_w=0.8):
        config = load_config()
        tokens = parse_api_tokens(api_tokens)
        
        if not tokens:
            raise Exception("Please provide at least one valid API Token")
        
        # Save new Token (if changed)
        if api_tokens and api_tokens.strip() != "" and not api_tokens.strip().startswith("***Saved"):
            if save_api_tokens(tokens):
                print(f"✅ Saved {len(tokens)} API Tokens")
            else:
                print("⚠️ API Token saving failed, but it doesn't affect current usage")
        
        print(f"🔍 Starting image generation...")
        print(f"📝 Prompt: {prompt}")
        print(f"❌ Negative Prompt: {negative_prompt if negative_prompt else 'None'}")
        print(f"🤖 Model: {model}")
        print(f"🔑 Available Token count: {len(tokens)}")
        print(f"📐 Size: {width}x{height}")
        print(f"🔄 Steps: {steps}")
        print(f"🧭 Guidance: {guidance}")
        print(f"🔢 Seed: {seed if seed != -1 else 'Random'}")
        
        # Print LoRA info
        lora_info = []
        if lora1_id.strip():
            lora_info.append(f"LoRA1: {lora1_id} (Weight: {lora1_w})")
        if lora2_id.strip():
            lora_info.append(f"LoRA2: {lora2_id} (Weight: {lora2_w})")
        if lora3_id.strip():
            lora_info.append(f"LoRA3: {lora3_id} (Weight: {lora3_w})")
        if lora_info:
            print(f"🔧 LoRA Config: {', '.join(lora_info)}")
        else:
            print("🔧 LoRA not used")
        
        last_exception = None
        for i, token in enumerate(tokens):
            try:
                print(f"🔄 Attempting to use token {i+1}/{len(tokens)}...")
                
                url = 'https://api-inference.modelscope.cn/v1/images/generations'
                payload = {
                    'model': model,
                    'prompt': prompt,
                    'size': f"{width}x{height}",
                    'steps': steps,
                    'guidance': guidance
                }
                
                lora_dict = {}
                if lora1_id and lora1_id.strip() != "":
                    lora_dict[lora1_id.strip()] = float(lora1_w)
                if lora2_id and lora2_id.strip() != "":
                    lora_dict[lora2_id.strip()] = float(lora2_w)
                if lora3_id and lora3_id.strip() != "":
                    lora_dict[lora3_id.strip()] = float(lora3_w)
                
                if lora_dict:
                    payload['loras'] = lora_dict
                    first_lora_id = next(iter(lora_dict.keys()))
                    first_lora_w = next(iter(lora_dict.values()))
                    payload['lora'] = first_lora_id
                    payload['lora_weight'] = first_lora_w
                
                if negative_prompt.strip():
                    payload['negative_prompt'] = negative_prompt
                if seed != -1:
                    payload['seed'] = seed
                else:
                    import random
                    payload['seed'] = random.randint(0, 2147483647)
                    print(f"🎲 Randomly generated seed: {payload['seed']}")
                
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                    'X-ModelScope-Async-Mode': 'true',
                    'X-ModelScope-Task-Type': 'text-to-image-generation',
                    'X-ModelScope-Request-Params': json.dumps({'loras': lora_dict} if lora_dict else {})
                }
                
                print(f"🚀 Sending API request to {model}...")
                submission_response = requests.post(
                    url, 
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), 
                    headers=headers,
                    timeout=config.get("timeout", 60)
                )
                
                if submission_response.status_code == 400:
                    print("⚠️ Standard request parameters failed, attempting to simplify parameters...")
                    minimal_payload = {
                        'model': model,
                        'prompt': prompt
                    }
                    if lora_dict:
                        minimal_payload['loras'] = lora_dict
                        minimal_payload['lora'] = first_lora_id
                        minimal_payload['lora_weight'] = first_lora_w
                    
                    submission_response = requests.post(
                        url,
                        data=json.dumps(minimal_payload, ensure_ascii=False).encode('utf-8'),
                        headers=headers,
                        timeout=config.get("timeout", 60)
                    )
                
                if submission_response.status_code != 200:
                    raise Exception(f"API request failed: {submission_response.status_code}, {submission_response.text}")
                
                submission_json = submission_response.json()
                image_url = None
                
                if 'task_id' in submission_json:
                    task_id = submission_json['task_id']
                    print(f"📌 Task ID obtained: {task_id}, starting to poll results...")
                    poll_start = time.time()
                    max_wait_seconds = max(60, config.get('timeout', 720))
                    while True:
                        task_resp = requests.get(
                            f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                            headers={
                                'Authorization': f'Bearer {token}',
                                'X-ModelScope-Task-Type': 'image_generation'
                            },
                            timeout=config.get("image_download_timeout", 120)
                        )
                        
                        if task_resp.status_code != 200:
                            raise Exception(f"Task query failed: {task_resp.status_code}, {task_resp.text}")
                        
                        task_data = task_resp.json()
                        status = task_data.get('task_status')
                        print(f"⌛ Task status: {status} (waited {int(time.time() - poll_start)} seconds)")
                        
                        if status == 'SUCCEED':
                            output_images = task_data.get('output_images') or []
                            if not output_images:
                                raise Exception("Task succeeded but no image URL returned")
                            image_url = output_images[0]
                            print(f"✅ Task completed, image URL obtained")
                            break
                        if status == 'FAILED':
                            raise Exception(f"Task failed: {task_data}")
                        if time.time() - poll_start > max_wait_seconds:
                            raise Exception(f"Task polling timed out ({max_wait_seconds}s), please try again later or reduce concurrency")
                        time.sleep(5)
                elif 'images' in submission_json and len(submission_json['images']) > 0:
                    image_url = submission_json['images'][0]['url']
                    print(f"✅ Image URL obtained directly")
                else:
                    raise Exception(f"Unrecognized API response format: {submission_json}")
                
                print(f"📥 Downloading image...")
                img_response = requests.get(image_url, timeout=config.get("image_download_timeout", 30))
                if img_response.status_code != 200:
                    raise Exception(f"Image download failed: {img_response.status_code}")
                
                print(f"🖼️ Processing image data...")
                pil_image = Image.open(BytesIO(img_response.content))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                print(f"✅ Token {i+1} call successful, image generation complete!")
                return (image_tensor,)
                
            except Exception as e:
                last_exception = e
                print(f"❌ Token {i+1} call failed: {str(e)}")
                if i < len(tokens) - 1:
                    print(f"⏳ Preparing to try next token...")
                    continue
                else:
                    break
        
        raise Exception(f"All {len(tokens)} API Tokens failed. Last error: {str(last_exception)}")

# -------------------------- Edit Node --------------------------
class ModelScopeImageEditNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        
        edit_models = config.get("image_edit_models", ["Qwen/Qwen-Image-Edit"])
        gen_models = config.get("image_models", ["Qwen/Qwen-Image"])

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Modify the content in the image"
                }),
                "api_tokens": ("STRING", {
                    "default": "***Saved {} tokens***".format(len(saved_tokens)) if saved_tokens else "",
                    "placeholder": "Please enter API Token (supports multiple, separated by comma/newline)" if not saved_tokens else "Leave blank to use saved tokens",
                    "multiline": True
                }),
                "image_gen_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Img2Img Mode",
                    "label_off": "Image Edit Mode"
                }),
            },
            "optional": {
                "gen_model": (gen_models, {
                    "default": gen_models[0] if gen_models else "Qwen/Qwen-Image"
                }),
                "edit_model": (edit_models, {
                    "default": edit_models[0] if edit_models else "Qwen/Qwen-Image-Edit"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                # LoRA related parameters
                "lora1_id": ("STRING", {"default": "", "label": "LoRA1 Model ID"}),
                "lora1_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA1 Weight"}),
                "lora2_id": ("STRING", {"default": "", "label": "LoRA2 Model ID"}),
                "lora2_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA2 Weight"}),
                "lora3_id": ("STRING", {"default": "", "label": "LoRA3 Model ID"}),
                "lora3_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA3 Weight"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "ModelScopeAPI"

    def edit_image(self, image, prompt, api_tokens, image_gen_mode=False, gen_model="Qwen/Qwen-Image", 
                   edit_model="Qwen/Qwen-Image-Edit", negative_prompt="", 
                   width=512, height=512, steps=30, guidance=3.5, seed=-1,
                   lora1_id="", lora1_w=0.8, lora2_id="", lora2_w=0.8, lora3_id="", lora3_w=0.8):
        config = load_config()
        tokens = parse_api_tokens(api_tokens)
        
        if not tokens:
            raise Exception("Please provide at least one valid API Token")
        
        # Save new Token (if changed)
        if api_tokens and api_tokens.strip() != "" and not api_tokens.strip().startswith("***Saved"):
            if save_api_tokens(tokens):
                print(f"✅ Saved {len(tokens)} API Tokens")
            else:
                print("⚠️ API Token saving failed, but it doesn't affect current usage")
        
        mode = "Img2Img Mode" if image_gen_mode else "Image Edit Mode"
        model = gen_model if image_gen_mode else edit_model
        
        print(f"🔍 Starting image editing...")
        print(f"📝 Prompt: {prompt}")
        print(f"❌ Negative Prompt: {negative_prompt if negative_prompt else 'None'}")
        print(f"🤖 Model: {model} ({mode})")
        print(f"🔑 Available Token count: {len(tokens)}")
        print(f"📐 Size: {width}x{height}")
        print(f"🔄 Steps: {steps}")
        print(f"🧭 Guidance: {guidance}")
        print(f"🔢 Seed: {seed if seed != -1 else 'Random'}")
        
        # Print LoRA info
        lora_info = []
        if lora1_id.strip():
            lora_info.append(f"LoRA1: {lora1_id} (Weight: {lora1_w})")
        if lora2_id.strip():
            lora_info.append(f"LoRA2: {lora2_id} (Weight: {lora2_w})")
        if lora3_id.strip():
            lora_info.append(f"LoRA3: {lora3_id} (Weight: {lora3_w})")
        if lora_info:
            print(f"🔧 LoRA Config: {', '.join(lora_info)}")
        else:
            print("🔧 LoRA not used")

        last_exception = None
        for i, token in enumerate(tokens):
            try:
                print(f"🔄 Attempting to use token {i+1}/{len(tokens)}...")
                
                temp_img_path = None
                image_url = None
                try:
                    # Save temporary image and upload
                    temp_img_path = os.path.join(tempfile.gettempdir(), f"qwen_edit_temp_{int(time.time())}.jpg")
                    if len(image.shape) == 4:
                        img = image[0]
                    else:
                        img = image
                    
                    img_np = 255. * img.cpu().numpy()
                    img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                    img_pil.save(temp_img_path)
                    print(f"💾 Saved temporary image to {temp_img_path}")
                    
                    # Upload image
                    upload_url = 'https://ai.kefan.cn/api/upload/local'
                    with open(temp_img_path, 'rb') as img_file:
                        files = {'file': img_file}
                        upload_response = requests.post(
                            upload_url,
                            files=files,
                            timeout=30
                        )
                        if upload_response.status_code == 200:
                            upload_data = upload_response.json()
                            if upload_data.get('success') == True and 'data' in upload_data:
                                image_url = upload_data['data']
                                print(f"📤 Image uploaded successfully, URL: {image_url[:50]}...")
                except Exception as e:
                    print(f"⚠️ Image upload failed, will use base64 encoding: {str(e)}")
                
                # Build request payload
                if not image_url:
                    print("🔄 Converting image to base64 format...")
                    image_data = tensor_to_base64_url(image)
                    payload = {
                        'model': model,
                        'prompt': prompt,
                        'image': image_data
                    }
                else:
                    payload = {
                        'model': model,
                        'prompt': prompt,
                        'image_url': image_url
                    }
                
                # Build LoRA parameters
                lora_dict = {}
                if lora1_id and lora1_id.strip() != "":
                    lora_dict[lora1_id.strip()] = float(lora1_w)
                if lora2_id and lora2_id.strip() != "":
                    lora_dict[lora2_id.strip()] = float(lora2_w)
                if lora3_id and lora3_id.strip() != "":
                    lora_dict[lora3_id.strip()] = float(lora3_w)
                
                if lora_dict:
                    payload['loras'] = lora_dict
                    first_lora_id = next(iter(lora_dict.keys()))
                    first_lora_w = next(iter(lora_dict.values()))
                    payload['lora'] = first_lora_id
                    payload['lora_weight'] = first_lora_w
                
                # Add other parameters
                if negative_prompt.strip():
                    payload['negative_prompt'] = negative_prompt
                if width != 512 or height != 512:
                    payload['size'] = f"{width}x{height}"
                if steps != 30:
                    payload['steps'] = steps
                if guidance != 3.5:
                    payload['guidance'] = guidance
                if seed != -1:
                    payload['seed'] = seed
                else:
                    import random
                    payload['seed'] = random.randint(0, 2147483647)
                    print(f"🎲 Randomly generated seed: {payload['seed']}")
                
                # Set headers
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                    'X-ModelScope-Async-Mode': 'true',
                    'X-ModelScope-Task-Type': 'image-to-image-generation',
                    'X-ModelScope-Request-Params': json.dumps({'loras': lora_dict} if lora_dict else {})
                }
                
                print(f"🚀 Sending API request to {model}...")
                url = 'https://api-inference.modelscope.cn/v1/images/generations'
                submission_response = requests.post(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers=headers,
                    timeout=config.get("timeout", 60)
                )
                
                if submission_response.status_code != 200:
                    raise Exception(f"API request failed: {submission_response.status_code}, {submission_response.text}")
                
                submission_json = submission_response.json()
                result_image_url = None
                
                if 'task_id' in submission_json:
                    task_id = submission_json['task_id']
                    print(f"📌 Task ID obtained: {task_id}, starting to poll results...")
                    poll_start = time.time()
                    max_wait_seconds = max(60, config.get('timeout', 720))
                    
                    while True:
                        task_resp = requests.get(
                            f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                            headers={
                                'Authorization': f'Bearer {token}',
                                'X-ModelScope-Task-Type': 'image_generation'
                            },
                            timeout=config.get("image_download_timeout", 120)
                        )
                        
                        if task_resp.status_code != 200:
                            raise Exception(f"Task query failed: {task_resp.status_code}, {task_resp.text}")
                        
                        task_data = task_resp.json()
                        status = task_data.get('task_status')
                        print(f"⌛ Task status: {status} (waited {int(time.time() - poll_start)} seconds)")
                        
                        if status == 'SUCCEED':
                            output_images = task_data.get('output_images') or []
                            if not output_images:
                                raise Exception("Task succeeded but no image URL returned")
                            result_image_url = output_images[0]
                            print(f"✅ Task completed, image URL obtained")
                            break
                        if status == 'FAILED':
                            error_message = task_data.get('errors', {}).get('message', 'Unknown error')
                            error_code = task_data.get('errors', {}).get('code', 'Unknown error code')
                            raise Exception(f"Task failed: Error code {error_code}, Error message: {error_message}")
                        if time.time() - poll_start > max_wait_seconds:
                            raise Exception(f"Task polling timed out ({max_wait_seconds}s), please try again later or reduce concurrency")
                        time.sleep(5)
                else:
                    raise Exception(f"Unrecognized API response format: {submission_json}")
                
                print(f"📥 Downloading edited image...")
                img_response = requests.get(result_image_url, timeout=config.get("image_download_timeout", 30))
                if img_response.status_code != 200:
                    raise Exception(f"Image download failed: {img_response.status_code}")
                
                print(f"🖼️ Processing image data...")
                pil_image = Image.open(BytesIO(img_response.content))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                # Cleanup temporary file
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                        print(f"🧹 Temporary image file deleted")
                    except:
                        print(f"⚠️ Unable to delete temporary image file {temp_img_path}")
                
                print(f"✅ Token {i+1} call successful, image editing complete!")
                return (image_tensor,)
                
            except Exception as e:
                last_exception = e
                print(f"❌ Token {i+1} call failed: {str(e)}")
                # Cleanup temporary file
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                if i < len(tokens) - 1:
                    print(f"⏳ Preparing to try next token...")
                    continue
                else:
                    break
        
        raise Exception(f"All {len(tokens)} API Tokens failed. Last error: {str(last_exception)}")

# -------------------------- Node Mapping --------------------------
NODE_CLASS_MAPPINGS = {
    "ModelScopeImageNode": ModelScopeImageNode,
    "ModelScopeImageEditNode": ModelScopeImageEditNode,
    "ModelScopeLoraPresetNode": ModelScopeLoraPresetNode,
    "ModelScopeSingleLoraLoaderNode": ModelScopeSingleLoraLoaderNode,
    "ModelScopeMultiLoraLoaderNode": ModelScopeMultiLoraLoaderNode
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageNode": "ModelScope Image Generation",
    "ModelScopeImageEditNode": "ModelScope Image Editing",
    "ModelScopeLoraPresetNode": "ModelScope LoRA Preset Management",
    "ModelScopeSingleLoraLoaderNode": "ModelScope LoRA Single Loader",
    "ModelScopeMultiLoraLoaderNode": "ModelScope LoRA Multi Loader"
}