# ComfyUI-LatentSyncWrapper

Unofficial [LatentSync](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on windows.

This node provides lip-sync capabilities in ComfyUI using ByteDance's LatentSync model. It allows you to synchronize video lips with audio input.

![image](https://github.com/user-attachments/assets/20971cd3-27c8-472e-92e9-afb95201bd23)

Add Kokoro option Workflow:
![image](https://github.com/user-attachments/assets/dd3a1de3-eca5-4c11-9c18-80750d464424)

https://github.com/user-attachments/assets/7a46a0dd-30d3-41d1-97c0-a56998636a28

## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working
2. Python 3.8-3.11 (mediapipe is not yet compatible with Python 3.12)
3. FFmpeg installed on your system:
- Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip) to your root C:\ drive, extract it, and add 'C:\ffmpeg\bin' to system PATH
4. If you get PYTHONPATH errors, ake sure Python is in your system PATH
     
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
diffusers
transformers
huggingface-hub
omegaconf
einops
opencv-python
mediapipe>=0.10.8
face-alignment
decord
ffmpeg-python
safetensors
soundfile
```
## Model Setup

The models can be obtained in two ways:

### Option 1: Automatic Download (First Run)
The node will attempt to automatically download required model files from HuggingFace on first use.
If automatic download fails, use Option 2.

### Option 2: Manual Download
1. Visit the HuggingFace repo: https://huggingface.co/chunyu-li/LatentSync
2. Download these files:
   - `latentsync_unet.pt`
   - `whisper/tiny.pt`
3. Place them in the following structure:
```bash
ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/
├── latentsync_unet.pt
└── whisper/
    └── tiny.pt
```
## Usage

1. Select an input video file
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. Connect to the LatentSync node
5. Run the workflow

The processed video will be saved in ComfyUI's output directory.

### Node Parameters:
- `video_path`: Path to input video file
- `audio`: Audio input from AceNodes audio loader
- `seed`: Random seed for reproducible results (default: 1247)


## Known Limitations

- Works best with clear, frontal face videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the video

### NEW - Video Length Adjuster Node
A complementary node that helps manage video length and synchronization with audio.

#### Features:
- Displays video and audio duration information
- Three modes of operation:
  - `normal`: Passes through video frames with added padding to prevent frame loss
  - `pingpong`: Creates a forward-backward loop of the video sequence
  - `loop_to_audio`: Extends video by repeating frames to match audio duration
  - `silent_padding_sec`: Adjast video length to audio

#### Usage:
1. Place the Video Length Adjuster between your video input and the LatentSync node
2. Connect audio to both the Video Length Adjuster and Video Combine nodes
3. Select desired mode based on your needs:
   - Use `normal` for standard lip-sync
   - Use `pingpong` for back-and-forth animation
   - Use `loop_to_audio` to match longer audio durations
   - Use `silent_padding_sec`to adjast longer video durations
#### Example Workflow:
1. Load Video (Upload) → Video frames output
2. Load Audio → Audio output
3. Connect both to Video Length Adjuster
4. Video Length Adjuster → LatentSync Node
5. LatentSync Node + Original Audio → Video Combine

## Troubleshooting

### mediapipe Installation Issues
If you encounter mediapipe installation errors:
1. Ensure you're using Python 3.8-3.11 (Check with `python --version`)
2. If using Python 3.12, you'll need to downgrade to a compatible version
3. Try installing mediapipe separately first:
   ```bash
   pip install mediapipe>=0.10.8

## Credits

This is an unofficial implementation based on:
- [LatentSync](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
