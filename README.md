# ComfyUI-ModelScope-API
 
ComfyUI-ModelScope-API is a powerful custom node for ComfyUI that bridges the gap between ComfyUI's visual workflow environment and ModelScope's extensive collection of AI models.

---
## Update Log:

**2026-01-01 Update** Updated support for Qwen-Image-Edit-2511 and Qwen-Image-2512.

**2025-12-12 Update** Added random seed selection to image analysis and text description nodes; now different descriptions can be generated each time.

**2025-12-12 Update** Fixed errors in ModelScope-Vision Image-to-Text node and resolved parsing errors in Qwen-VL model's ModelScope image description node caused by spelling mistakes (Thanks to ShmilyStar for feedback).

**2025-12-11 Update** Image generation models now support Z-image models. Referenced lora code from https://github.com/otsluo/comfyui-modelscope-api to add LoRA functionality to image generation and image editing nodes. Image description nodes can now use Qwen3-VL as a large language model directly without inputting an image.

**2025-10-23 Update** Added a secondary prompt input to the ModelScope-API image description node, allowing 2 prompts to be entered simultaneously (automatically merged).

**2025-10-22 Update** Added image description node supporting Qwen3-VL series model reverse prompting.

## 🏗️ Core Architecture Overview
 
The core of this architecture is the `ModelScopeAPI` class, which serves as the primary interface between ComfyUI and ModelScope cloud services. The architecture adopts a modular design pattern with clear separation of responsibilities, handling everything from user input validation to image processing and API communication.
 
### Architectural Features
 
- **Modular Design**: Clear separation of responsibilities, easy to maintain and extend.
- **Unified Interface**: All model calls are handled through a single API class.
- **Cloud Processing**: No need to download models locally; generate images directly in the cloud.
- **Parameter Validation**: Comprehensive input parameter validation and error handling.
 
---
 
### ✨ Features
 
- **Multi-Model Support**: Freely enter model names; supports all compatible models under ModelScope.
- **Dual-Mode Generation**:
  - **Text-to-Image Mode**: Generate images solely through text prompts.
  - **Image-to-Image Mode**: Transform images based on input images and text prompts.
- **Direct API Calls**: No need to download models locally; generate images directly in the cloud via API.
- **Full Parameter Control**: Supports adjusting resolution (width and height), random seed, sampling steps, and prompt guidance scale (Guidance).
- **Built-in Image Hosting**: Automatically uploads input images to free image hosting to obtain public URLs for API use.
- **Flexible Model Selection**: Freely enter any ModelScope-supported model name in the UI.
 
---
 
## 🖼️ Usage Examples
 
### Text-to-Image Mode
 
[Text Prompt] → [ModelScope API Node] → [Generated Image]
 
### Image-to-Image Mode
 
[Input Image] + [Text Prompt] → [ModelScope API Node] → [Transformed Image]
 
---
 
## ⚙️ Installation
 
### Method 1: Using Git
 
1. Open a terminal or command line window.
2. Navigate to the `custom_nodes` folder in your ComfyUI installation directory.
   ```bash
   cd /path/to/your/ComfyUI/custom_nodes/
   ```
3. Run the following command to clone this repository:
   ```bash
   git clone https://github.com/hujuying/ComfyUI-ModelScope-API.git
   ```

### Method 2: Manual Download
1. Click the **Code** button at the top right of this page and select **Download ZIP**.
2. Extract the downloaded ZIP file.
3. Move the extracted folder (ensure the folder name is `ComfyUI-ModelScope-API`) to the `custom_nodes` directory of ComfyUI.
4. Restart ComfyUI.

### 🚀 How to Use
### General Steps
1. In ComfyUI, add this node through the right-click menu or by double-clicking and searching for `ModelScope`.
2. In the `api_key` field, enter your ModelScope API Key.
3. In the `model_name` field, enter the name of the model you want to use (e.g., `MusePublic/FLUX.1-Kontext-Dev`).
4. In the `prompt` field, enter your desired image description or modification prompt.
5. Adjust other parameters as needed.
6. Connect the node's `IMAGE` output to a `PreviewImage` or `SaveImage` node to view the result.

### Text-to-Image Mode
- Do not connect any image to the node's `image` input.
- Provide only a text prompt; the node will automatically perform a text-to-image operation.

### Image-to-Image Mode
- Connect an image output to this node's `image` input.
- Provide a text prompt describing the changes you want to make to the image.
- The node will perform an image-to-image operation based on the input image.

### 📋 Parameter Description

| Parameter | Type | Range | Use |
| :--- | :--- | :--- | :--- |
| api_key | String | - | Authentication for ModelScope API access |
| model_name | String | - | Identifier of the target model |
| prompt | String | - | Text description used for generation |
| image | Image (Optional) | - | Input image for image-to-image mode |
| width | Integer | 64-2048 | Output image width (pixels) |
| height | Integer | 64-2048 | Output image height (pixels) |
| seed | Integer | 0-2147483647 | Random seed for reproducible results |
| steps | Integer | 1-100 | Number of sampling steps |
| guidance | Float | 1.5-20.0 | Prompt guidance scale |

### 🎯 Supported Models

### Officially Supported Models
| Model Name | Description | Use Case |
| :--- | :--- | :--- |
| MusePublic/FLUX.1-Kontext-Dev | Default FLUX model for general image generation | Text-to-Image, Image-to-Image |
| MusePublic/Qwen-image | Specialized for detailed scenes | Complex composition and details |
| MusePublic/Qwen-Image-Edit | Specialized model for image editing | Image modification and enhancement |
| MusePublic/489_ckpt_FLUX_1 | FLUX series variant | High-quality image generation |
| MAILAND/majicflus_v1 | MajicFlus model | Artistic style generation |
...and other compatible models on the ModelScope platform.

### 💡 Best Practices
**Prompt Enhancement**
Use a prompt enhancement node before the ModelScope API to improve your text descriptions.

**Post-Processing**
Connect the output to image enhancement or upscaling nodes for final refinement.

**Batch Processing**
Create multiple ModelScope API nodes with different seeds to generate variants of the same concept.

**Image-to-Image Prompt Tips**
For image-to-image generation, prompts should describe the changes you want to make, rather than the entire image. For example, if you input a photo of a cat, a prompt like "make it look like a watercolor painting" will be more effective than "watercolor style cat".

### 🔑 How to Obtain a ModelScope Access Token
1. **Login or Register**: Visit https://www.modelscope.ai/. If you already have an account, click the "Login" button in the upper right corner. If not, click "Register" to create an account.
2. **Go to Personal Homepage**: After logging in, hover over your avatar in the upper right corner.
3. **Access Tokens Page**: In the dropdown menu, click "Access Tokens".
4. **View or Generate Token**: On the "Access Tokens" page, you can see your personal Access Token. If you haven't generated one before, the system may prompt you to generate a new token.

### 🙏 Acknowledgments
- API Service Provider: [ModelScope](https://www.modelscope.ai/)
- Model Provider: MusePublic
- Image Upload Service: freeimage.host

📄 **License**
This project is open-sourced under the MIT License.
