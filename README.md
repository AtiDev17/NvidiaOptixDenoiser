# NvidiaOptixDenoiser

A simple, high-performance command-line tool to denoise images (EXR, PNG, JPG) using NVIDIA's AI-accelerated OptiX Denoiser.

## Features
- **AI-Powered**: Uses NVIDIA OptiX 9+ for superior quality.
- **Tiled Denoising**: Handles large resolutions (4K, 8K+) automatically or manually to fit in VRAM.
- **Guide Layers**: Supports Albedo (Color) and Normal maps for better detail preservation.
- **Fast**: GPU-accelerated via CUDA.

## Requirements
- **NVIDIA GPU**: Maxwell architecture or newer (RTX recommended).
- **Drivers**: NVIDIA GeForce Driver 590.00+ / NVIDIA Studio Driver 590.00+.
- **CUDA**: CUDA Toolkit 13.x (only).
- **OptiX**: OptiX SDK 9.1 or newer.
- **Python**: 3.8 or newer.
- **Libraries**: `cupy`, `numpy`, `Pillow`, and the `optix` binding.

## Installation

1. **Get the Script**
   Clone this repository:
   ```bash
   git clone https://github.com/AtiDev17/NvidiaOptixDenoiser.git
   cd NvidiaOptixDenoiser
   ```

2. **Install Python Dependencies**
   ```bash
   It is recommended to create a virtual environment before installing the modules.
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Denoise a noisy image and save the result:
```bash
python denoiser.py render_noisy.exr output_clean.png
```

### Improving Quality (Feature Guides)
For the best quality, provide the "Albedo" (Raw Color) and "Normal" passes from your renderer. This helps the AI distinguish between textures and noise.

```bash
python denoiser.py render_noisy.exr output_clean.png --albedo render_albedo.exr --normal render_normal.exr
```
*Note: Normal maps should ideally be in Camera Space.*

### Handling Large Images (Tiling)
The script automatically handles tiling to prevent "Out of Memory" errors on large images. If you want to force a specific tile size (e.g., 1024x1024):
```bash
python denoiser.py input.png output.png --tile-size 1024
```

## Arguments

| Argument | Description |
| :--- | :--- |
| `input` | Path to the noisy input image (EXR, PNG, JPG, TIF). |
| `output` | Path where the denoised image will be saved. |
| `--albedo` | (Optional) Path to the Albedo/Diffuse color pass. |
| `--normal` | (Optional) Path to the Normal pass. |
| `--tile-size` | (Optional) Size of tiles in pixels. Default is `0` (Auto-calculate based on VRAM). |

## Notes
- **HDR Support**: `.exr` files are recommended for input/output to preserve high dynamic range data.
