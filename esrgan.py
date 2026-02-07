"""
Self-contained Real-ESRGAN upscaler for anime content.

Implements the full RRDBNet architecture and a tiled inference wrapper so that
there are zero external dependencies beyond PyTorch and PIL.  This avoids the
version conflicts that crop up with the realesrgan / basicsr packages.

Supported models:
  - RealESRGAN_x4plus_anime_6B  (RRDBNet, 6 RRDB blocks, 4× scale)

The state-dict key names exactly match the official weights file so you can
load the .pth directly with strict=True.

Usage:
    from esrgan import load_esrgan_upscaler

    upscaler = load_esrgan_upscaler("weights/RealESRGAN_x4plus_anime_6B.pth")
    big_image = upscaler.upscale(small_image)   # PIL Image → PIL Image (4×)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    """Five-conv residual dense block with LeakyReLU."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3 × RDB)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet — the backbone of Real-ESRGAN.

    Default parameters match ``RealESRGAN_x4plus_anime_6B.pth``::

        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=6, num_grow_ch=32, scale=4
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 6,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Two 2× nearest-neighbour upsample stages = 4× total
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# Tiled inference wrapper
# ---------------------------------------------------------------------------

class TiledUpscaler:
    """Process images of arbitrary size by splitting into overlapping tiles.

    Each tile is padded, run through the model, then the padding is cropped
    from the output before stitching.  This keeps GPU memory bounded regardless
    of input resolution.
    """

    def __init__(
        self,
        model: nn.Module,
        scale: int = 4,
        tile_size: int = 512,
        tile_pad: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def upscale(self, img: Image.Image) -> Image.Image:
        """Upscale a PIL Image by ``self.scale``× and return a new PIL Image."""
        img_np = np.array(img).astype(np.float32) / 255.0
        # Handle RGBA → RGB
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, self.dtype)

        _, c, h, w = tensor.shape
        out_h, out_w = h * self.scale, w * self.scale
        output = torch.zeros(1, c, out_h, out_w, device=self.device, dtype=self.dtype)

        ts = self.tile_size
        pad = self.tile_pad
        s = self.scale

        for y0 in range(0, h, ts):
            for x0 in range(0, w, ts):
                # Tile region (may be smaller than ts at image edges)
                x1 = min(x0 + ts, w)
                y1 = min(y0 + ts, h)

                # Padded input region (clamped to image bounds)
                px0 = max(x0 - pad, 0)
                py0 = max(y0 - pad, 0)
                px1 = min(x1 + pad, w)
                py1 = min(y1 + pad, h)

                tile_in = tensor[:, :, py0:py1, px0:px1]
                tile_out = self.model(tile_in)

                # Crop the padding from the upscaled output
                crop_y0 = (y0 - py0) * s
                crop_x0 = (x0 - px0) * s
                crop_y1 = crop_y0 + (y1 - y0) * s
                crop_x1 = crop_x0 + (x1 - x0) * s

                output[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = \
                    tile_out[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

        result = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_esrgan_upscaler(
    weights_path: str,
    *,
    scale: int = 4,
    num_block: int = 6,
    tile_size: int = 512,
    tile_pad: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> TiledUpscaler:
    """Load Real-ESRGAN weights and return a ready-to-use TiledUpscaler.

    Args:
        weights_path: Path to ``.pth`` weights file.
        scale: Upscaling factor (4 for RealESRGAN_x4plus_anime_6B).
        num_block: Number of RRDB blocks (6 for the anime model).
        tile_size: Processing tile size in pixels.
        tile_pad: Overlap padding per tile edge.
        device: Torch device string.
        dtype: Inference dtype (float16 recommended for speed).

    Returns:
        A :class:`TiledUpscaler` instance.
    """
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=num_block, num_grow_ch=32,
        scale=scale,
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Some releases wrap the actual weights under a key
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device, dtype)

    print(f"ESRGAN loaded: {weights_path}  ({num_block} blocks, {scale}× scale, "
          f"tile={tile_size}, pad={tile_pad})")

    return TiledUpscaler(
        model, scale=scale, tile_size=tile_size,
        tile_pad=tile_pad, device=device, dtype=dtype,
    )
