#!/usr/bin/env python3
"""
Multi-stage SDXL image generator with LoRA support, ESRGAN upscaling,
and sliding-window tiled diffusion refinement.

Modes:
    python generate.py                     # full pipeline, continuous
    python generate.py --once              # full pipeline, single run
    python generate.py --stage1            # stage 1 only, continuous (for curation)
    python generate.py --stage1 --once     # stage 1 only, single run
    python generate.py --upscale           # upscale all pending stage1 outputs

Workflow for curation:
    1. python generate.py --stage1         # generate many base images
    2. Delete unwanted images from outputs/stage1/
    3. python generate.py --upscale        # upscale the survivors

Output structure:
    outputs/
    ├── stage1/           base images (curate these)
    ├── metadata/         JSON run configs (kept for --upscale)
    ├── final/            end results
    └── intermediates/    stage2, upscaled, tiles
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        desc = kw.get("desc", "")
        total = kw.get("total", None)
        for i, x in enumerate(it):
            if total:
                print(f"\r  {desc}: {i+1}/{total}", end="", flush=True)
            yield x
        print()

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.set_num_threads(os.cpu_count() or 4)
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda"

# All relative paths resolve against the script's own directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Recursive dot-notation wrapper over a YAML dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __repr__(self) -> str:
        return f"Config({vars(self)})"

    def to_dict(self) -> dict:
        return {
            k: v.to_dict() if isinstance(v, Config) else v
            for k, v in vars(self).items()
        }

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))


def get_dtype(cfg: Config) -> torch.dtype:
    dt = getattr(cfg.performance, "dtype", "bf16")
    return torch.bfloat16 if dt == "bf16" else torch.float16


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_seed(s: int) -> int:
    if isinstance(s, int) and s >= 0:
        return s
    return int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF


def sanitize(s: str, maxlen: int = 140) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = s.replace(".", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if s else "none"


def read_text(path: str, fallback: str = "") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return fallback


def round8(n: int) -> int:
    """Round down to nearest multiple of 8 (VAE latent compatibility)."""
    return (n // 8) * 8


def resolve_path(p: str) -> str:
    """Resolve relative paths against SCRIPT_DIR."""
    return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt handling
# ═══════════════════════════════════════════════════════════════════════════

def load_prompt_options(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.split(","):
                part = part.strip()
                if part:
                    items.append(part)
    random.shuffle(items)
    return items


def pick_extras(options: List[str], kmax: int) -> List[str]:
    if not options or kmax <= 0:
        return []
    k = random.randint(0, min(kmax, len(options)))
    return random.sample(options, k) if k > 0 else []


def build_prompt(base: str, additions: List[str]) -> str:
    additions = [p.strip() for p in additions if p and p.strip()]
    return base + ", " + ", ".join(additions) if additions else base


# ═══════════════════════════════════════════════════════════════════════════
# LoRA discovery
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LoRAInfo:
    kind: str
    path: str
    name: str
    adapter: str
    triggers: List[str] = field(default_factory=list)


def _parse_triggers(txt_path: str) -> List[str]:
    s = read_text(txt_path).strip()
    return [p.strip() for p in s.split(",") if p.strip()] if s else []


def discover_loras(folder: str, kind: str) -> List[LoRAInfo]:
    folder = os.path.abspath(resolve_path(folder))
    if not os.path.isdir(folder):
        print(f"  LoRA dir not found: {folder}")
        return []
    files = sorted(set(
        glob.glob(os.path.join(folder, "*.safetensors"))
        + glob.glob(os.path.join(folder, "*.safetensor"))
    ))
    result: List[LoRAInfo] = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        txt = os.path.join(os.path.dirname(path), name + ".txt")
        triggers = _parse_triggers(txt) if os.path.exists(txt) else []
        adapter = f"{kind}__{sanitize(name, 60)}"
        result.append(LoRAInfo(kind=kind, path=path, name=name,
                               adapter=adapter, triggers=triggers))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ESRGAN upscaler wrapper
# ═══════════════════════════════════════════════════════════════════════════

def load_upscaler(cfg: Config, dtype: torch.dtype):
    """Load ESRGAN if configured, else None (Lanczos fallback)."""
    method = getattr(cfg.upscaler, "method", "lanczos")
    if method != "esrgan":
        print("Upscaler: Lanczos (ESRGAN not selected)")
        return None

    model_path = resolve_path(cfg.upscaler.model_path)
    if not os.path.exists(model_path):
        print(f"WARNING: ESRGAN weights not found at {model_path}")
        print("  Download: https://github.com/xinntao/Real-ESRGAN/releases/"
              "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
        print("  Falling back to Lanczos.")
        return None

    try:
        if SCRIPT_DIR not in sys.path:
            sys.path.insert(0, SCRIPT_DIR)
        from esrgan import load_esrgan_upscaler
        return load_esrgan_upscaler(
            model_path,
            tile_size=getattr(cfg.upscaler, "tile_size", 512),
            tile_pad=getattr(cfg.upscaler, "tile_pad", 32),
            dtype=dtype,
        )
    except Exception as e:
        print(f"WARNING: ESRGAN load failed ({e}), falling back to Lanczos.")
        return None


def upscale_image(img: Image.Image, target_w: int, target_h: int,
                  esrgan=None) -> Image.Image:
    """Upscale to exact (target_w, target_h). ESRGAN 4x then resize, or Lanczos."""
    if esrgan is not None:
        upscaled = esrgan.upscale(img)
        if upscaled.size != (target_w, target_h):
            upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        return upscaled
    return img.resize((target_w, target_h), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline loading
# ═══════════════════════════════════════════════════════════════════════════

def _needs_refiner(cfg: Config, mode: str) -> bool:
    if mode == "stage1":
        return getattr(cfg.model, "use_refiner", False)
    s2 = cfg.generation.stage2
    s3 = cfg.generation.stage3
    needs_later = (
        (getattr(s2, "enabled", False) and getattr(s2, "use_refiner", False))
        or (getattr(s3, "enabled", False) and getattr(s3, "use_refiner", False))
    )
    if mode == "upscale":
        return needs_later
    return getattr(cfg.model, "use_refiner", False) or needs_later


def _configure_pipe(pipe, strategy: str):
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    if strategy == "gpu":
        pipe.to(DEVICE)
    elif strategy == "sequential_offload":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()


def _try_compile(pipe, cfg: Config, label: str):
    if not getattr(cfg.performance, "compile_unet", False):
        return

    # Triton (required by the inductor backend) is not available on Windows
    if sys.platform == "win32":
        print(f"  torch.compile skipped for {label} (Triton not available on Windows)")
        print(f"    Tip: use WSL2 for torch.compile support")
        return

    mode = getattr(cfg.performance, "compile_mode", "default")
    try:
        pipe.unet = torch.compile(pipe.unet, mode=mode)
        print(f"  torch.compile({label} UNet, mode={mode})")
    except Exception as e:
        print(f"  torch.compile({label}) failed: {e}, continuing without")


def load_pipelines(
    cfg: Config, all_loras: List[LoRAInfo], mode: str, dtype: torch.dtype,
) -> Tuple[Any, Any, Optional[Any], List[LoRAInfo], bool, set]:
    """Load pipelines appropriate for the run mode.

    Returns (t2i, i2i, refiner, loaded_loras, shared_flag, loaded_adapter_names).
    t2i or i2i may be None if not needed for the mode.
    """
    strategy = getattr(cfg.memory, "strategy", "model_offload")
    need_t2i = mode in ("full", "stage1")
    need_i2i = mode in ("full", "upscale")
    need_ref = _needs_refiner(cfg, mode)

    t2i = None
    i2i = None
    refiner = None
    shared = False

    if need_t2i:
        print(f"Loading base t2i: {cfg.model.base}")
        t2i = StableDiffusionXLPipeline.from_pretrained(cfg.model.base, torch_dtype=dtype)
        _try_compile(t2i, cfg, "t2i")

    if need_i2i:
        if t2i is not None and strategy == "gpu":
            print("Building i2i from shared components (gpu mode)...")
            i2i = StableDiffusionXLImg2ImgPipeline(
                vae=t2i.vae, text_encoder=t2i.text_encoder,
                text_encoder_2=t2i.text_encoder_2, unet=t2i.unet,
                tokenizer=t2i.tokenizer, tokenizer_2=t2i.tokenizer_2,
                scheduler=t2i.scheduler,
            )
            shared = True
        else:
            print(f"Loading i2i pipeline ({strategy})...")
            i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                cfg.model.base, torch_dtype=dtype
            )
            _try_compile(i2i, cfg, "i2i")

    if need_ref:
        print(f"Loading refiner: {cfg.model.refiner}")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            cfg.model.refiner, torch_dtype=dtype
        )
        _try_compile(refiner, cfg, "refiner")

    for pipe in [t2i, i2i, refiner]:
        if pipe is not None:
            _configure_pipe(pipe, strategy)

    # --- LoRA loading ---
    loaded: List[LoRAInfo] = []
    loaded_adapters: set = set()

    for lora in all_loras:
        if not os.path.exists(lora.path):
            continue
        try:
            if t2i is not None:
                t2i.load_lora_weights(lora.path, adapter_name=lora.adapter)
            if i2i is not None and not shared:
                i2i.load_lora_weights(lora.path, adapter_name=lora.adapter)
            loaded.append(lora)
            loaded_adapters.add(lora.adapter)
        except Exception as e:
            print(f"  LoRA load failed: {lora.path} — {e}")

    if loaded:
        kinds = {l.kind for l in loaded}
        print(f"Loaded {len(loaded)} LoRAs ({', '.join(sorted(kinds))})")
    else:
        print("WARNING: no LoRAs loaded.")

    return t2i, i2i, refiner, loaded, shared, loaded_adapters


def apply_loras(t2i, i2i, adapters: List[Tuple[str, float]], shared: bool):
    """Activate the given LoRA adapters on whichever pipelines exist."""
    if not adapters:
        return
    names = [a for a, _ in adapters]
    weights = [float(w) for _, w in adapters]
    if t2i is not None:
        t2i.set_adapters(names, adapter_weights=weights)
    if i2i is not None and not shared:
        i2i.set_adapters(names, adapter_weights=weights)


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion wrappers
# ═══════════════════════════════════════════════════════════════════════════

def run_t2i(t2i, refiner, prompt: str, neg: str, seed_val: int,
            w: int, h: int, gen_cfg: Config) -> Image.Image:
    gen = torch.Generator(device=DEVICE).manual_seed(seed_val)
    steps = gen_cfg.t2i.steps
    cfg_scale = gen_cfg.t2i.cfg
    split = gen_cfg.t2i.denoise_split

    if refiner is not None:
        lat = t2i(
            prompt=prompt, negative_prompt=neg,
            width=w, height=h,
            num_inference_steps=steps, guidance_scale=cfg_scale,
            denoising_end=split, output_type="latent", generator=gen,
        ).images
        return refiner(
            prompt=prompt, negative_prompt=neg, image=lat,
            num_inference_steps=steps, guidance_scale=cfg_scale,
            denoising_start=split, generator=gen,
        ).images[0]

    return t2i(
        prompt=prompt, negative_prompt=neg,
        width=w, height=h,
        num_inference_steps=steps, guidance_scale=cfg_scale,
        generator=gen,
    ).images[0]


def run_i2i(i2i, refiner, prompt: str, neg: str, seed_val: int,
            init_img: Image.Image, stage_cfg: Config,
            use_refiner: bool = True) -> Image.Image:
    gen = torch.Generator(device=DEVICE).manual_seed(seed_val)
    steps = stage_cfg.steps
    cfg_scale = stage_cfg.cfg
    strength = stage_cfg.strength
    split = getattr(stage_cfg, "denoise_split", 0.8)

    if refiner is not None and use_refiner:
        lat = i2i(
            prompt=prompt, negative_prompt=neg,
            image=init_img, strength=strength,
            num_inference_steps=steps, guidance_scale=cfg_scale,
            denoising_end=split, output_type="latent", generator=gen,
        ).images
        return refiner(
            prompt=prompt, negative_prompt=neg, image=lat,
            num_inference_steps=steps, guidance_scale=cfg_scale,
            denoising_start=split, generator=gen,
        ).images[0]

    return i2i(
        prompt=prompt, negative_prompt=neg,
        image=init_img, strength=strength,
        num_inference_steps=steps, guidance_scale=cfg_scale,
        generator=gen,
    ).images[0]


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window tiled refinement
# ═══════════════════════════════════════════════════════════════════════════

def compute_tiles(width: int, height: int, tile_size: int,
                  overlap: int) -> List[Tuple[int, int, int, int]]:
    stride = tile_size - overlap
    assert stride > 0, f"tile_size ({tile_size}) must exceed overlap ({overlap})"

    xs = list(range(0, width - tile_size, stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    xs = sorted(set(xs))

    ys = list(range(0, height - tile_size, stride))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    ys = sorted(set(ys))

    return [(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]


def _blend_mask(
    tw: int, th: int, overlap: int,
    x0: int, y0: int, x1: int, y1: int,
    img_w: int, img_h: int,
    mode: str = "cosine",
) -> np.ndarray:
    mask = np.ones((th, tw), dtype=np.float32)
    ov = min(overlap, tw // 2, th // 2)
    if ov <= 0:
        return mask

    if mode == "cosine":
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, ov, dtype=np.float32)))
    else:
        ramp = np.linspace(0, 1, ov, dtype=np.float32)

    if x0 > 0:     mask[:, :ov]  *= ramp[None, :]
    if x1 < img_w: mask[:, -ov:] *= ramp[::-1][None, :]
    if y0 > 0:     mask[:ov, :]  *= ramp[:, None]
    if y1 < img_h: mask[-ov:, :] *= ramp[::-1][:, None]

    return mask


def refine_tiled(
    i2i, refiner, img: Image.Image,
    prompt: str, neg: str, seed_val: int,
    cfg: Config,
    save_dir: Optional[str] = None,
    tag: str = "",
) -> Image.Image:
    W, H = img.size
    s3 = cfg.generation.stage3
    tile_size = getattr(s3, "tile_size", 1024)
    overlap = getattr(s3, "tile_overlap", 256)
    use_ref = getattr(s3, "use_refiner", False)
    blend_mode = getattr(s3, "blend_mode", "cosine")

    assert tile_size % 8 == 0, f"tile_size must be multiple of 8, got {tile_size}"

    tiles = compute_tiles(W, H, tile_size, overlap)
    print(f"  Tiled refinement: {len(tiles)} tiles, {tile_size}px, "
          f"{overlap}px overlap, blend={blend_mode}")

    accum = np.zeros((H, W, 3), dtype=np.float64)
    weight = np.zeros((H, W, 1), dtype=np.float64)

    for idx, (x0, y0, x1, y1) in enumerate(
        tqdm(tiles, desc="  Tiles", unit="tile")
    ):
        crop = img.crop((x0, y0, x1, y1))
        tile_seed = seed_val + idx * 1337

        out = run_i2i(i2i, refiner, prompt, neg, tile_seed, crop, s3,
                      use_refiner=use_ref)

        tw, th = out.size
        mask = _blend_mask(tw, th, overlap, x0, y0, x1, y1, W, H, blend_mode)
        mask3 = mask[:, :, None]

        tile_np = np.array(out).astype(np.float64)
        accum[y0:y1, x0:x1] += tile_np * mask3
        weight[y0:y1, x0:x1] += mask3

        if save_dir:
            out.save(os.path.join(save_dir, f"{tag}_tile_{idx:03d}.png"))

    blended = accum / np.clip(weight, 1e-8, None)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════
# Output directories & resolutions
# ═══════════════════════════════════════════════════════════════════════════

def setup_dirs(cfg: Config) -> Dict[str, str]:
    base = resolve_path(cfg.output.dir)
    dirs = {
        "root":       base,
        "stage1":     os.path.join(base, "stage1"),
        "stage1_done": os.path.join(base, "stage1_done"),
        "metadata":   os.path.join(base, "metadata"),
        "final":      os.path.join(base, "final"),
        "stage2_up":  os.path.join(base, "intermediates", "stage2_up"),
        "stage2":     os.path.join(base, "intermediates", "stage2"),
        "stage3_up":  os.path.join(base, "intermediates", "stage3_up"),
        "tiles":      os.path.join(base, "intermediates", "tiles"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def get_resolutions(cfg: Config) -> Dict[str, Tuple[int, int]]:
    gen = cfg.generation
    bw = round8(gen.base_width)
    bh = round8(gen.base_height)
    return {
        "base":  (bw, bh),
        "mid":   (round8(bw * gen.mid_scale), round8(bh * gen.mid_scale)),
        "final": (round8(bw * gen.final_scale), round8(bh * gen.final_scale)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: generate base image
# ═══════════════════════════════════════════════════════════════════════════

def choose_lora(loaded: List[LoRAInfo], kind: str) -> Optional[LoRAInfo]:
    pool = [l for l in loaded if l.kind == kind]
    return random.choice(pool) if pool else None


def do_stage1(
    t2i, refiner, loaded_loras: List[LoRAInfo], shared: bool,
    options: List[str], cfg: Config, dirs: Dict[str, str],
    res: Dict[str, Tuple[int, int]],
) -> Optional[str]:
    """Generate one stage-1 image. Returns the tag, or None on error."""

    base_prompt = read_text(
        resolve_path(cfg.prompts.prompt_file), "masterpiece, best quality"
    )
    neg = read_text(
        resolve_path(cfg.prompts.negative_prompt_file),
        "worst quality, low quality, blurry, jpeg artifacts, watermark, signature",
    )

    # --- LoRA selection ---
    char = choose_lora(loaded_loras, "character")
    style = choose_lora(loaded_loras, "style")

    adapters: List[Tuple[str, float]] = []
    triggers: List[str] = []
    lora_meta: List[dict] = []

    for lora in [char, style]:
        if lora is not None:
            strength = (cfg.lora.character_strength if lora.kind == "character"
                        else cfg.lora.style_strength)
            adapters.append((lora.adapter, strength))
            triggers += lora.triggers
            lora_meta.append({
                "kind": lora.kind,
                "name": lora.name,
                "path": lora.path,
                "adapter": lora.adapter,
                "strength": strength,
                "triggers": lora.triggers,
            })

    apply_loras(t2i, None, adapters, shared)

    # --- Build final prompt ---
    # Triggers first, then base prompt, then extras
    extras = pick_extras(options, cfg.prompts.max_extras)
    parts = triggers + [base_prompt] + extras
    prompt = ", ".join(p.strip() for p in parts if p and p.strip())

    # --- Seed & tag ---
    seed_val = resolve_seed(cfg.generation.seed)
    tag = stamp()
    bw, bh = res["base"]

    # --- Generate ---
    print(f"\n  [{tag}] Stage 1: {bw}x{bh}  seed={seed_val}")
    img = run_t2i(t2i, refiner, prompt, neg, seed_val, bw, bh, cfg.generation)

    # --- Save image ---
    img_path = os.path.join(dirs["stage1"], f"{tag}_stage1_{bw}x{bh}.png")
    img.save(img_path)
    print(f"  -> {img_path}")

    # --- Save metadata ---
    meta = {
        "tag": tag,
        "seed": seed_val,
        "prompt": prompt,
        "negative_prompt": neg,
        "prompt_components": {
            "base": base_prompt,
            "triggers": triggers,
            "extras": extras,
        },
        "loras": lora_meta,
        "resolution": {"base": [bw, bh]},
        "stage1": {
            "steps": cfg.generation.t2i.steps,
            "cfg": cfg.generation.t2i.cfg,
            "denoise_split": cfg.generation.t2i.denoise_split,
            "used_refiner": refiner is not None,
        },
        "files": {
            "stage1": os.path.basename(img_path),
        },
    }
    meta_path = os.path.join(dirs["metadata"], f"{tag}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    c = char.name if char else "none"
    s = style.name if style else "none"
    print(f"  char={c}  style={s}  extras={extras}")

    return tag


# ═══════════════════════════════════════════════════════════════════════════
# Later stages: upscale from a stage-1 image
# ═══════════════════════════════════════════════════════════════════════════

def do_later_stages(
    img_base: Image.Image, tag: str, meta: dict,
    i2i, refiner, esrgan,
    loaded_adapters: set, shared: bool,
    cfg: Config, dirs: Dict[str, str],
    res: Dict[str, Tuple[int, int]],
) -> str:
    """Run stages 2, 3, and final upscale. Returns final image path."""
    prompt = meta["prompt"]
    neg = meta["negative_prompt"]
    seed_val = meta["seed"]

    gen = cfg.generation
    save_inter = getattr(cfg.output, "save_intermediates", True)
    mw, mh = res["mid"]
    fw, fh = res["final"]

    # --- Activate LoRAs from metadata ---
    adapters: List[Tuple[str, float]] = []
    for lora_info in meta.get("loras", []):
        adapter = lora_info["adapter"]
        strength = lora_info["strength"]
        if adapter in loaded_adapters:
            adapters.append((adapter, strength))
        else:
            print(f"  WARNING: LoRA '{lora_info['name']}' not available, skipping")
    apply_loras(None, i2i, adapters, shared)

    # --- Stage 2 ---
    img_current = img_base
    if gen.stage2.enabled:
        print(f"  Stage 2: upscale -> {mw}x{mh}, img2img refine")
        up_mid = upscale_image(img_current, mw, mh, esrgan=None)
        if save_inter:
            p = os.path.join(dirs["stage2_up"],
                             f"{tag}_stage2_up_{mw}x{mh}.png")
            up_mid.save(p)

        s2_use_ref = getattr(gen.stage2, "use_refiner", False)
        img_current = run_i2i(i2i, refiner, prompt, neg, seed_val + 9001,
                              up_mid, gen.stage2, use_refiner=s2_use_ref)
        if save_inter:
            p = os.path.join(dirs["stage2"],
                             f"{tag}_stage2_{mw}x{mh}.png")
            img_current.save(p)
            print(f"  -> {p}")
    else:
        img_current = upscale_image(img_current, mw, mh)

    # --- Final upscale ---
    print(f"  {'ESRGAN' if esrgan else 'Lanczos'} upscale -> {fw}x{fh}")
    up_final = upscale_image(img_current, fw, fh, esrgan=esrgan)

    if save_inter:
        p = os.path.join(dirs["stage3_up"],
                         f"{tag}_stage3_up_{fw}x{fh}.png")
        up_final.save(p)

    # --- Stage 3 tiled refinement ---
    if gen.stage3.enabled:
        final = refine_tiled(
            i2i, refiner, up_final, prompt, neg, seed_val + 18001, cfg,
            save_dir=dirs["tiles"] if save_inter else None,
            tag=tag,
        )
    else:
        final = up_final

    # --- Save final ---
    final_path = os.path.join(dirs["final"], f"{tag}_final_{fw}x{fh}.png")
    final.save(final_path)
    print(f"  ★ {final_path}")

    # --- Move stage1 source to done folder ---
    s1_name = meta.get("files", {}).get("stage1", "")
    if s1_name:
        s1_src = os.path.join(dirs["stage1"], s1_name)
        s1_dst = os.path.join(dirs["stage1_done"], s1_name)
        if os.path.exists(s1_src):
            shutil.move(s1_src, s1_dst)
            print(f"  Moved stage1 -> stage1_done/")

    return final_path


# ═══════════════════════════════════════════════════════════════════════════
# Upscale mode: process pending stage-1 outputs
# ═══════════════════════════════════════════════════════════════════════════

def find_pending(dirs: Dict[str, str], res: Dict[str, Tuple[int, int]]) -> List[dict]:
    """Find JSONs whose stage1 image still exists but final output doesn't."""
    fw, fh = res["final"]
    pending = []

    meta_dir = dirs["metadata"]
    if not os.path.isdir(meta_dir):
        return []

    for jf in sorted(glob.glob(os.path.join(meta_dir, "*.json"))):
        tag = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        # Stage1 image must still exist (not deleted by user)
        s1_name = meta.get("files", {}).get("stage1", "")
        s1_path = os.path.join(dirs["stage1"], s1_name)
        if not os.path.exists(s1_path):
            continue

        # Final must not already exist
        final_path = os.path.join(dirs["final"], f"{tag}_final_{fw}x{fh}.png")
        if os.path.exists(final_path):
            continue

        pending.append({
            "tag": tag,
            "meta": meta,
            "stage1_path": s1_path,
        })

    return pending


def run_upscale_mode(
    i2i, refiner, esrgan,
    loaded_adapters: set, shared: bool,
    cfg: Config, dirs: Dict[str, str],
    res: Dict[str, Tuple[int, int]],
):
    """Process all pending stage-1 images through later stages."""
    pending = find_pending(dirs, res)

    if not pending:
        print("\nNo pending images to upscale.")
        print("  (need stage1 images with matching metadata, and no existing final)")
        return

    print(f"\nFound {len(pending)} pending image(s):\n")
    for i, item in enumerate(pending, 1):
        print(f"  {i}. {item['tag']}")
    print()

    done = 0
    for i, item in enumerate(pending, 1):
        tag = item["tag"]
        meta = item["meta"]

        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(pending)}] {tag}")
        print(f"{'─' * 60}")

        try:
            img_base = Image.open(item["stage1_path"]).convert("RGB")
            do_later_stages(
                img_base, tag, meta,
                i2i, refiner, esrgan,
                loaded_adapters, shared,
                cfg, dirs, res,
            )
            done += 1
        except Exception:
            print(f"  ERROR processing {tag}:")
            traceback.print_exc()

    print(f"\nUpscale complete. {done}/{len(pending)} processed successfully.")


# ═══════════════════════════════════════════════════════════════════════════
# Full mode: stage 1 + later stages in one pass
# ═══════════════════════════════════════════════════════════════════════════

def do_full_run(
    t2i, i2i, refiner, esrgan,
    loaded_loras: List[LoRAInfo],
    loaded_adapters: set, shared: bool,
    options: List[str],
    cfg: Config, dirs: Dict[str, str],
    res: Dict[str, Tuple[int, int]],
):
    bw, bh = res["base"]
    fw, fh = res["final"]

    print(f"\n{'═' * 64}")
    print(f"  Full run: {bw}x{bh} -> {fw}x{fh}")
    print(f"{'═' * 64}")

    tag = do_stage1(t2i, refiner, loaded_loras, shared, options, cfg, dirs, res)
    if tag is None:
        return

    meta_path = os.path.join(dirs["metadata"], f"{tag}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    s1_name = meta["files"]["stage1"]
    img_base = Image.open(os.path.join(dirs["stage1"], s1_name)).convert("RGB")

    do_later_stages(
        img_base, tag, meta,
        i2i, refiner, esrgan,
        loaded_adapters, shared,
        cfg, dirs, res,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage SDXL image generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  (default)   full pipeline: stage 1 + upscale, continuous\n"
            "  --stage1    generate base images only (for curation)\n"
            "  --upscale   upscale pending stage1 outputs (after curation)\n"
            "\n"
            "workflow:\n"
            "  1. python generate.py --stage1          # generate many\n"
            "  2. delete unwanted images from outputs/stage1/\n"
            "  3. python generate.py --upscale          # upscale survivors\n"
        ),
    )
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="Config YAML file (default: config.yaml)")
    parser.add_argument("--stage1", action="store_true",
                        help="Generate stage 1 images only")
    parser.add_argument("--upscale", action="store_true",
                        help="Upscale all pending stage 1 outputs")
    parser.add_argument("--once", action="store_true",
                        help="Single run then exit (ignored for --upscale)")
    args = parser.parse_args()

    if args.stage1 and args.upscale:
        parser.error("--stage1 and --upscale are mutually exclusive")

    mode = "stage1" if args.stage1 else "upscale" if args.upscale else "full"

    # --- Config ---
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = resolve_path(config_path)
    if not os.path.exists(config_path):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(config_path)
    dtype = get_dtype(cfg)
    dirs = setup_dirs(cfg)
    res = get_resolutions(cfg)

    bw, bh = res["base"]
    fw, fh = res["final"]
    dt_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    print(f"\nMode: {mode}  |  dtype: {dt_name}  |  {bw}x{bh} -> {fw}x{fh}")

    # --- Prompt options (not needed for upscale mode) ---
    options: List[str] = []
    if mode != "upscale":
        options = load_prompt_options(resolve_path(cfg.prompts.options_file))
        if not options:
            print(f"NOTE: no prompt options from {cfg.prompts.options_file}")

    # --- LoRA discovery ---
    print("Discovering LoRAs...")
    chars = discover_loras(cfg.lora.character_dir, "character")
    styles = discover_loras(cfg.lora.style_dir, "style")
    print(f"  Found: {len(chars)} character, {len(styles)} style")
    all_loras = chars + styles
    if not all_loras:
        print("ERROR: no LoRA files found.")
        sys.exit(1)

    # --- ESRGAN (not needed for stage1-only) ---
    esrgan = None
    if mode != "stage1":
        esrgan = load_upscaler(cfg, dtype)

    # --- Diffusion pipelines (mode-aware) ---
    t2i, i2i, refiner, loaded, shared, loaded_adapters = load_pipelines(
        cfg, all_loras, mode, dtype
    )
    if not loaded:
        print("ERROR: no LoRAs loaded.")
        sys.exit(1)

    if mode != "upscale":
        if not any(l.kind == "character" for l in loaded):
            print("WARNING: no character LoRAs loaded.")
        if not any(l.kind == "style" for l in loaded):
            print("WARNING: no style LoRAs loaded.")

    # --- Dispatch ---
    if mode == "upscale":
        run_upscale_mode(
            i2i, refiner, esrgan,
            loaded_adapters, shared,
            cfg, dirs, res,
        )
        return

    label = "Stage 1 only" if mode == "stage1" else "Full pipeline"
    loop = "single run" if args.once else "continuous (Ctrl+C to stop)"
    print(f"\n{label}, {loop}\n")

    count = 0
    try:
        while True:
            try:
                if mode == "stage1":
                    do_stage1(t2i, refiner, loaded, shared, options,
                              cfg, dirs, res)
                else:
                    do_full_run(t2i, i2i, refiner, esrgan,
                                loaded, loaded_adapters, shared,
                                options, cfg, dirs, res)
                count += 1
                print(f"  (#{count} complete)")

            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

            if args.once:
                break
    except KeyboardInterrupt:
        print(f"\nStopped after {count} run(s).")
        sys.exit(0)


if __name__ == "__main__":
    main()
