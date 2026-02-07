# SDXL Image Generator

Three-mode SDXL image generation pipeline with LoRA support, ESRGAN upscaling, and tiled diffusion refinement.

**Key Features:**
- 🎯 LoRA support for character and style
- 🚀 Three flexible generation modes (full pipeline, stage 1 only, batch upscale)
- 📐 Multi-stage upscaling: base → mid → final resolution
- 🎨 Optional tiled diffusion refinement for final quality
- 💾 Curated workflow: generate many, keep the best, then upscale
- ⚡ Memory-efficient with multiple strategies (4GB–16GB VRAM)

## Three Generation Modes

### Mode 1: Full Pipeline (Default)
```bash
python generate.py                    # continuous generation
python generate.py --once             # single run
```
Generates images through all enabled stages in sequence. Each image goes from base generation through to final output automatically.

### Mode 2: Stage 1 Only (Curation Workflow)
```bash
python generate.py --stage1           # continuous generation
python generate.py --stage1 --once    # single run
```
Generates base-resolution images only and saves metadata. Perfect for generating many candidates quickly, then manually curating before committing to expensive upscaling.

### Mode 3: Batch Upscale (Process Curated Images)
```bash
python generate.py --upscale
```
Finds all stage 1 images that still exist in `outputs/stage1/` with matching metadata in `outputs/metadata/`, then processes them through stages 2 and 3. Automatically skips images that already have final outputs.

### Recommended Curation Workflow
```bash
# 1. Generate many base images quickly
python generate.py --stage1

# 2. Review outputs/stage1/ and delete unwanted images

# 3. Upscale and refine the survivors
python generate.py --upscale
```

## Setup

### 1. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pyyaml pillow
pip install xformers  # optional but recommended for memory efficiency
pip install tqdm      # optional, for progress bars
```

### 2. Download ESRGAN Weights (Optional)
```bash
mkdir -p weights
wget -P weights/ https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```
If ESRGAN weights are missing, the pipeline automatically falls back to Lanczos resampling.

### 3. Add Your LoRAs

Place your LoRA files in the appropriate folders:
```
loras/
├── character/
│   ├── my_character.safetensors
│   └── my_character.txt          # optional: trigger words (comma-separated)
└── style/
    ├── my_style.safetensors
    └── my_style.txt
```

**LoRA Discovery:**
- Supports both `.safetensors` and `.safetensor` extensions
- Automatically discovers all LoRAs in both folders at startup
- Randomly selects one character and one style LoRA per generation
- Trigger words from `.txt` files are automatically prepended to prompts

### 4. Configure Prompts

Edit these text files to customize your prompts:

| File | Purpose |
|------|---------|
| `manual_prompt.txt` | Base positive prompt (always included) |
| `manual_negative_prompt.txt` | Negative prompt (always included) |
| `prompt_options.txt` | Pool of additional phrases (0–5 randomly sampled per generation) |

## Pipeline Stages

The generator uses a three-stage upscaling pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Text-to-Image (base resolution)                    │
│ • Default: 768×1344 (9:16 portrait)                         │
│ • Uses base SDXL model + selected LoRAs                     │
│ • Optional: refiner handoff at 80% denoising               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Upscale + Img2Img Refinement (mid resolution)     │
│ • Default: 2× scale → 1536×2688                            │
│ • Upscale method: Lanczos (ESRGAN not used here)          │
│ • Img2img at 25% strength with optional refiner           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: ESRGAN + Tiled Refinement (final resolution)      │
│ • Default: 4× scale → 3072×5376                            │
│ • ESRGAN 4× upscale (or Lanczos fallback)                  │
│ • Optional: sliding-window tiled diffusion refinement      │
│ • Tiles: 2048px with 256px overlap, cosine blending       │
└─────────────────────────────────────────────────────────────┘
```

Each stage can be enabled/disabled independently in `config.yaml`.

## Output Structure

```
outputs/
├── stage1/              # Base resolution images (curate these!)
├── metadata/            # JSON files with generation parameters
├── final/               # Final output images
└── intermediates/       # Optional intermediate saves
    ├── stage2_up/       # Mid-resolution upscaled (pre-refinement)
    ├── stage2/          # Mid-resolution refined
    ├── stage3_up/       # Final-resolution upscaled (pre-tiling)
    └── tiles/           # Individual diffusion tiles (if stage 3 enabled)
```

**File Naming Convention:**
All files use timestamp tags for easy matching:
```
20260207_143022_stage1_768x1344.png       # stage1/
20260207_143022.json                       # metadata/
20260207_143022_stage2_up_1536x2688.png   # intermediates/stage2_up/
20260207_143022_stage2_1536x2688.png      # intermediates/stage2/
20260207_143022_stage3_up_3072x5376.png   # intermediates/stage3_up/
20260207_143022_tile_000.png              # intermediates/tiles/
20260207_143022_final_3072x5376.png       # final/
```

Metadata JSON files contain complete generation parameters including prompt, seed, LoRA selections, and all stage settings.

## Configuration

### Quick Start: Common Settings

**Aspect Ratio** (use SDXL bucket sizes, ~1M pixels):
```yaml
generation:
  base_width: 768      # 9:16 portrait (YouTube Shorts)
  base_height: 1344
```

| Aspect | Width | Height | Use Case |
|--------|-------|--------|----------|
| 1:1    | 1024  | 1024   | Square, social media |
| 9:16   | 768   | 1344   | Vertical video, mobile |
| 16:9   | 1344  | 768    | Widescreen, landscape |
| 2:3    | 832   | 1216   | Portrait photo |
| 3:2    | 1216  | 832    | Landscape photo |

**Memory Strategy**:
```yaml
memory:
  strategy: "model_offload"  # good balance for 12GB GPUs
```

| Strategy | VRAM Usage | Speed | Best For |
|----------|------------|-------|----------|
| `gpu` | ~16GB+ | Fastest | High VRAM GPUs (shared UNet components) |
| `model_offload` | ~8GB | Good | **Recommended for 12GB GPUs** |
| `sequential_offload` | ~4GB | Slow | Minimum VRAM systems |

**Performance**:
```yaml
performance:
  dtype: "fp16"          # "bf16" for Ampere/Blackwell, "fp16" for wider compatibility
  compile_unet: false    # 20-40% speedup after first run (Linux only, slow first time)
```

### Advanced: Pipeline Control

**Enable/Disable Stages**:
```yaml
generation:
  stage2:
    enabled: true      # Set false to skip mid-resolution refinement
  stage3:
    enabled: true      # Set false to skip tiled refinement (faster, lower quality)
```

**Refiner Control**:
```yaml
model:
  use_refiner: true    # Stage 1 refiner handoff

generation:
  stage2:
    use_refiner: true  # Stage 2 refiner handoff
  stage3:
    use_refiner: true  # Stage 3 per-tile refinement (expensive!)
```

**Tiled Refinement Settings**:
```yaml
generation:
  stage3:
    tile_size: 2048        # Diffusion tile size (must be multiple of 8)
    tile_overlap: 256      # Overlap prevents seams
    blend_mode: "cosine"   # "cosine" (smooth) or "linear"
    strength: 0.4          # Img2img strength per tile
```

## File Descriptions

| File | Purpose |
|------|---------|
| `generate.py` | Main pipeline with three generation modes |
| `esrgan.py` | Self-contained RealESRGAN upscaler (no external deps beyond PyTorch) |
| `config.yaml` | All pipeline settings and hyperparameters |
| `manual_prompt.txt` | Base positive prompt |
| `manual_negative_prompt.txt` | Negative prompt |
| `prompt_options.txt` | Extra prompt phrases (randomly sampled) |
| `loras/character/` | Character LoRA files (`.safetensors` + optional `.txt`) |
| `loras/style/` | Style LoRA files (`.safetensors` + optional `.txt`) |

## Technical Details

### LoRA System
- Character LoRAs: Default strength 1.0
- Style LoRAs: Default strength 0.85
- One of each type randomly selected per generation
- Trigger words automatically prepended to prompt
- Adapters persist across generations for efficiency

### Memory Optimizations
- VAE slicing and tiling enabled
- xformers memory-efficient attention (if available)
- Model CPU offloading based on strategy
- Shared UNet components in GPU mode
- Optional torch.compile for 20–40% speedup (Linux only)

### ESRGAN Implementation
- Custom tiled upscaler (512px tiles, 32px padding)
- Supports RealESRGAN_x4plus_anime_6B model
- No external dependencies (pure PyTorch + PIL)
- Falls back to Lanczos if weights missing

### Scheduling
- UniPCMultistep scheduler for all stages
- Base/refiner denoising split at 80%
- Separate seed offsets per stage for variation

## Platform Notes

- **torch.compile**: Only works on Linux (requires Triton). Automatically skipped on Windows.
- **bf16**: Recommended for Ampere (RTX 30xx), Hopper (RTX 40xx), or Blackwell (RTX 50xx) GPUs
- **fp16**: Use for older GPUs or if bf16 causes issues

## Tips

1. **For fast iteration:** Use `--stage1 --once` to quickly test prompt/LoRA combinations
2. **For quality:** Enable all stages and use ESRGAN with tiled refinement
3. **For speed:** Disable stage 3 tiled refinement (set `stage3.enabled: false`)
4. **For VRAM:** Start with `model_offload` strategy, adjust up/down as needed
5. **For curation:** Generate 20-50 stage1 images, keep the best 5, then upscale

## License

This pipeline uses the following:
- Diffusers (Apache 2.0)
- SDXL base models (varies by model, check HuggingFace)
- RealESRGAN (BSD-3-Clause)

Check individual model licenses on HuggingFace before use.
