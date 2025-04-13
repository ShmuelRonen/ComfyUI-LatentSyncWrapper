# ComfyUI-LatentSyncWrapper 1.5

## Support My Work
If you find this project helpful, consider buying me a coffee:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=shmuelronen&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/shmuelronen)

Unofficial [LatentSync 1.5](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on Windows and WSL 2.0.

This node provides advanced lip-sync capabilities in ComfyUI using ByteDance's LatentSync 1.5 model. It allows you to synchronize video lips with audio input with improved temporal consistency and better performance on a wider range of languages.

![image](https://github.com/user-attachments/assets/85e4dafe-2adf-4994-9440-8a435a5ea6d8)


## What's new in LatentSync 1.5?

1. **Temporal Layer Improvements**: Corrected implementation now provides significantly improved temporal consistency compared to version 1.0
2. **Better Chinese Language Support**: Performance on Chinese videos is now substantially improved through additional training data
3. **Reduced VRAM Requirements**: Now only requires 20GB VRAM (can run on RTX 3090) through various optimizations:
   - Gradient checkpointing in U-Net, VAE, SyncNet and VideoMAE
   - Native PyTorch FlashAttention-2 implementation (no xFormers dependency)
   - More efficient CUDA cache management
   - Focused training of temporal and audio cross-attention layers only
4. **Code Optimizations**:
   - Removed dependencies on xFormers and Triton
   - Upgraded to diffusers 0.32.2

## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working

2. FFmpeg installed on your system:
   - Windows: Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add to system PATH

## Installation

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
```

## Note on Model Downloads

On first use, the node will automatically download required model files from HuggingFace:
- LatentSync 1.5 UNet model
- Whisper model for audio processing
- You can also manually download the models from HuggingFace repo: https://huggingface.co/ByteDance/LatentSync-1.5

### Checkpoint Directory Structure

After successful installation and model download, your checkpoint directory structure should look like this:

```
./checkpoints/
|-- .cache/
|-- auxiliary/
|-- whisper/
|   `-- tiny.pt
|-- config.json
|-- latentsync_unet.pt  (~5GB)
|-- stable_syncnet.pt   (~1.6GB)
```

Make sure all these files are present for proper functionality. The main model files are:
- `latentsync_unet.pt`: The primary LatentSync 1.5 model
- `stable_syncnet.pt`: The SyncNet model for lip-sync supervision
- `whisper/tiny.pt`: The Whisper model for audio processing

## Usage

1. Select an input video file with AceNodes video loader
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. (Optional) Adjust the lips_expression parameter to control lip movement intensity
5. (Optional) Modify the inference_steps parameter to balance quality and speed
6. Connect to the LatentSync1.5 node
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
- For speeches or presentations where clear lip movements are important, try increasing the lips_expression value to 2.0-2.5
- For casual conversations, the default value of 1.5 usually works well
- If lip movements appear unnatural or exaggerated, try lowering the lips_expression value
- Different values may work better for different languages and speech patterns
- If you need higher quality results and have time to wait, increase inference_steps to 30-50
- For quicker previews or less critical applications, reduce inference_steps to 10-15

### Common issue in portable version
For ComfyUI portable version users having this issue, here's how to fix the decord import error:
First, locate your Python executable in the portable version:
```
python.exe -m pip uninstall decord
python.exe -m pip install decord --no-cache-dir
or use python3 instead
```
## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video

## Credits

This is an unofficial implementation based on:
- [LatentSync 1.5](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
