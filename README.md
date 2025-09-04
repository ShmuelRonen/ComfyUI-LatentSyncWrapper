# ComfyUI-LatentSyncWrapper 1.6

## Support My Work
If you find this project helpful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=shmuelronen&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/shmuelronen)

Unofficial [LatentSync 1.6](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on Windows and WSL 2.0.

This node provides advanced lip-sync capabilities in ComfyUI using ByteDance's LatentSync 1.6 model. It allows you to synchronize video lips with audio input with enhanced clarity and resolution, addressing the blurriness issues found in previous versions.

## ⚠️ IMPORTANT UPGRADE NOTICE ⚠️

**If you have a previous version of ComfyUI-LatentSyncWrapper installed, you MUST completely remove it before installing version 1.6:**

1. **Stop ComfyUI** completely
2. **Delete the entire folder**: `ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/`
3. **Clean installation**: Follow the installation steps below for a fresh 1.6 installation
4. **Do NOT try to update** - version 1.6 requires a complete reinstallation due to significant changes

**Failure to remove the previous version will cause conflicts and prevent proper operation.**

![image](https://github.com/user-attachments/assets/85e4dafe-2adf-4994-9440-8a435a5ea6d8)

### Last Changes:
#### June-14-25 - Updated to LatentSync 1.6 with 512×512 resolution training for significantly improved clarity and detail in teeth and lip generation.
#### April-29-25 - To avoid GPU memory allocation issues, frames are now moved to CPU before uint8 conversion. This change enables generation of longer videos without OOM errors.

## What's new in LatentSync 1.6?

1. **Enhanced Resolution Training**: LatentSync 1.6 is trained on 512×512 resolution videos to address the blurriness issues reported in LatentSync 1.5
2. **Improved Visual Quality**: Significantly reduces blurriness in teeth and lips that was common in version 1.5
3. **Backward Compatibility**: The current code is compatible with both LatentSync 1.5 and 1.6, requiring only checkpoint changes
4. **Same Model Architecture**: No changes to model structure or training strategy - only upgraded training dataset resolution
5. **Maintained Performance**: All the improvements from version 1.5 are retained:
   - **Temporal Layer Improvements**: Corrected implementation provides significantly improved temporal consistency
   - **Better Chinese Language Support**: Enhanced performance on Chinese videos through additional training data
   - **Reduced VRAM Requirements**: Optimized to run on 20GB VRAM (RTX 3090 compatible)
   - **Code Optimizations**: Native PyTorch FlashAttention-2 implementation without xFormers dependency

## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working

2. FFmpeg installed on your system:
   - Windows: Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add to system PATH

## Installation

**Note**: A complete pre-configured checkpoints package is available via Google Drive (recommended), or you can download models individually from HuggingFace repositories.

Only proceed with installation after confirming all prerequisites are installed and working.

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
cd ComfyUI-LatentSyncWrapper
pip install -r requirements.txt
```

## Required Dependencies
```
diffusers>=0.32.2
transformers
huggingface-hub
omegaconf
einops
opencv-python
mediapipe
face-alignment
decord
ffmpeg-python
safetensors
soundfile
DeepCache
```

## Manual Model Download Required

**Important**: LatentSync 1.6 requires manual model downloads because the LatentSync 1.6 models are hosted on a private HuggingFace repository that cannot be automatically accessed. You must download the following models before first use:

### Manual Individual Downloads

### 1. VAE Model Download
Create a `vae` folder inside your `checkpoints` directory and download the VAE model:

```bash
# Create the vae directory
mkdir checkpoints/vae
```

**Manual download steps:**
1. Visit: https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main
2. Download **only these 2 files**:
   - `diffusion_pytorch_model.safetensors`
   - `config.json`
3. Place them in `checkpoints/vae/` folder (inside the extension directory)

### 2. LatentSync 1.6 Checkpoints Download
Download the main LatentSync 1.6 models:

```bash
# Download LatentSync 1.6 models from HuggingFace
# Visit: https://huggingface.co/ByteDance/LatentSync-1.6/tree/main
# Download all files from the repository into checkpoints/ folder
```

**Manual download steps:**
1. **Ensure you have access** to the private HuggingFace repository
2. Visit: https://huggingface.co/ByteDance/LatentSync-1.6/tree/main
3. Download all files from the repository
4. Place them directly in the `checkpoints/` folder

### Checkpoint Directory Structure

After downloading models (using either option), your checkpoint directory structure should look like this:

```
./checkpoints/
|-- .cache/
|-- auxiliary/
|-- vae/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- whisper/
|   `-- tiny.pt
|-- config.json
|-- latentsync_unet.pt  (~5GB)
|-- stable_syncnet.pt   (~1.6GB)
```

Make sure all these files are present for proper functionality. The main model files are:
- `vae/diffusion_pytorch_model.safetensors`: The Stable Diffusion VAE model for encoding/decoding
- `vae/config.json`: VAE configuration file
- `latentsync_unet.pt`: The primary LatentSync 1.6 model trained at 512×512 resolution
- `stable_syncnet.pt`: The SyncNet model for lip-sync supervision
- `whisper/tiny.pt`: The Whisper model for audio processing

## Usage

1. Select an input video file with AceNodes video loader
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. (Optional) Adjust the lips_expression parameter to control lip movement intensity
5. (Optional) Modify the inference_steps parameter to balance quality and speed
6. Connect to the LatentSync1.6 node
7. Run the workflow

The processed video will be saved in ComfyUI's output directory.

### Node Parameters:
- `video_path`: Path to input video file
- `audio`: Audio input from AceNodes audio loader
- `seed`: Random seed for reproducible results (default: 1247)
- `lips_expression`: Controls the expressiveness of lip movements (default: 1.5)
  - Higher values (2.0-3.0): More pronounced lip movements, better for expressive speech
  - Lower values (1.0-1.5): Subtler lip movements, better for calm speech
  - This parameter affects the model's guidance scale, balancing between natural movement and lip sync accuracy
- `inference_steps`: Number of denoising steps during inference (default: 20)
  - Higher values (30-50): Better quality results but slower processing
  - Lower values (10-15): Faster processing but potentially lower quality
  - The default of 20 usually provides a good balance between quality and speed

### Tips for Better Results:
- **Enhanced 512×512 Resolution**: LatentSync 1.6 provides significantly clearer teeth and lip details compared to version 1.5
- For speeches or presentations where clear lip movements are important, try increasing the lips_expression value to 2.0-2.5
- For casual conversations, the default value of 1.5 usually works well
- If lip movements appear unnatural or exaggerated, try lowering the lips_expression value
- Different values may work better for different languages and speech patterns
- If you need higher quality results and have time to wait, increase inference_steps to 30-50
- For quicker previews or less critical applications, reduce inference_steps to 10-15

## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video
- **Requires manual model downloads** - LatentSync 1.6 models are hosted on a private HuggingFace repository, but a complete package is available via Google Drive
- Individual model downloads require access to the ByteDance/LatentSync-1.6 HuggingFace repository

## Credits

This is an unofficial implementation based on:
- [LatentSync 1.6](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
