"""
inference.py — AquaVision AI model wrapper.

Keeps the model architecture + loading + inference logic isolated from the API layer.

The HTTP API uses enhance(): native-resolution U-Net with bounded float tile buffers.
enhance_hd() and super-resolution remain experimental and are not exposed by the API.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

try:                                   # imported as a package: model.inference / backend.model.inference
    from .hd_pipeline import hd_enhance, run_tiled_image, to_uint8
except ImportError:                    # run as a plain script: `python main.py` inside model/
    from hd_pipeline import hd_enhance, run_tiled_image, to_uint8

CHECKPOINT_PATH = Path(__file__).parent / "best.pt"
logger = logging.getLogger(__name__)
BASE_CHANNELS = 32
DEPTH = 3
DROPOUT = 0.1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth
        channels: List[int] = [base_channels * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch
        self.pool = nn.MaxPool2d(2)

        bottleneck_ch = channels[-1] * 2
        self.bottleneck = nn.Sequential(ConvBlock(channels[-1], bottleneck_ch), nn.Dropout2d(dropout))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_ch
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
            prev_ch = ch

        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            h = self._match_and_concat(h, skip)
            h = dec(h)
        return torch.sigmoid(self.out_conv(h))

    @staticmethod
    def _match_and_concat(upsampled, skip):
        if upsampled.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[-2] - upsampled.shape[-2]
            diff_w = skip.shape[-1] - upsampled.shape[-1]
            skip = skip[..., diff_h // 2: skip.shape[-2] - (diff_h - diff_h // 2),
                             diff_w // 2: skip.shape[-1] - (diff_w - diff_w // 2)]
        return torch.cat([upsampled, skip], dim=1)


class AquaVisionModel:
    """
    Loads once, reused across requests. Instantiate a single instance at API startup —
    never per-request, that would reload weights on every call and be far too slow.
    """

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, device: str | None = None,
                 tile_size: int | None = None):
        checkpoint_path = Path(checkpoint_path)
        requested_device = device or os.getenv("AQUAVISION_DEVICE", "cpu")
        if requested_device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("AQUAVISION_DEVICE must be cpu, cuda, mps or auto.")
        threads = int(os.getenv("TORCH_NUM_THREADS", "1"))
        if threads < 1:
            raise ValueError("TORCH_NUM_THREADS must be positive.")
        torch.set_num_threads(threads)
        self.tile_size = int(tile_size if tile_size is not None else os.getenv("INFERENCE_TILE_SIZE", "384"))
        if not 64 < self.tile_size <= 384 or self.tile_size % 8:
            raise ValueError("INFERENCE_TILE_SIZE must be a multiple of 8 between 72 and 384.")
        if requested_device != "auto":
            self.device = requested_device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = UNet(base_channels=BASE_CHANNELS, depth=DEPTH, dropout=DROPOUT)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}'. "
                f"Place best.pt there or pass a different path."
            )

        # Load trusted local tensors on CPU without executing pickle globals.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint.get("epoch")
        with checkpoint_path.open("rb") as source:
            source_hash = checkpoint.get("source_sha256") or hashlib.file_digest(source, "sha256").hexdigest()
        self.model_version = f"unet-feather-v2-t{self.tile_size}-o64-{source_hash[:12]}"
        del checkpoint  # Release optimizer/training tensors before serving requests.
        self.model.to(self.device)
        self.model.eval()

        # Optional super-resolution stage (see load_super_resolution)
        self.sr_model = None
        self.sr_scale = 1
        logger.info("model_loaded device=%s epoch=%s model_version=%s", self.device, self.epoch, self.model_version)

    # ------------------------------------------------------------------ super-resolution (optional)
    def load_super_resolution(self, weights_path: Path) -> None:
        """
        Loads a Real-ESRGAN style .pth (e.g. realesr-general-x4v3.pth or RealESRGAN_x4plus.pth) via `spandrel`.
        Raises if `spandrel` is not installed or the weights are missing / not an image-to-image model.
        """
        from spandrel import ImageModelDescriptor, ModelLoader

        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Super-resolution weights not found at '{weights_path}'.")
        descriptor = ModelLoader().load_from_file(str(weights_path))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"'{weights_path}' is not an image super-resolution model.")
        descriptor.to(self.device).eval()
        self.sr_model = descriptor
        self.sr_scale = int(descriptor.scale)
        print(f"[AquaVisionModel] super-resolution x{self.sr_scale} loaded from {Path(weights_path).name}")

    # ------------------------------------------------------------------ numpy <-> torch glue
    @torch.no_grad()
    def _predict_np(self, rgb: np.ndarray) -> np.ndarray:
        """U-Net on one HxWx3 float32 [0,1] array (any size). Pads to a multiple of 2**depth, crops back."""
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        _, _, h, w = x.shape
        multiple = 2 ** self.model.depth
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"   # reflect needs pad < size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        y = self.model(x)[:, :, :h, :w]
        return y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def _sr_predict_np(self, rgb: np.ndarray) -> np.ndarray:
        """Super-resolution on one HxWx3 float32 [0,1] tile -> (H*scale)x(W*scale)x3."""
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        y = self.sr_model(x)
        return y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    # ------------------------------------------------------------------ public API
    def enhance_hd(
        self,
        img: Image.Image,
        work_long_side: int = 768,
        dehaze_strength: float = 0.0,
        sr_max_input_long_side: int = 1280,
        sr_target_long_side: int = 1920,
        keep_original_size: bool = True,
        sharpen_amount: float = 0.0,
    ) -> Image.Image:
        """
        Experimental pipeline. Applies the phone's EXIF rotation (so original and enhanced always match),
        fixes colour with the U-Net (keeping the original detail), removes haze, and — if a super-resolution
        model is loaded — cleans/sharpens small images (same size) or upscales them to ``sr_target_long_side``.
        """
        img = ImageOps.exif_transpose(img).convert("RGB")
        rgb = np.asarray(img, dtype=np.float32) / 255.0

        out = hd_enhance(
            rgb,
            self._predict_np,
            work_long_side=work_long_side,
            dehaze_strength=dehaze_strength,
            sr_predict_fn=self._sr_predict_np if self.sr_model is not None else None,
            sr_scale=self.sr_scale,
            sr_max_input_long_side=sr_max_input_long_side,
            sr_target_long_side=sr_target_long_side,
            keep_original_size=keep_original_size,
            sharpen_amount=sharpen_amount,
        )
        return Image.fromarray(to_uint8(out))

    @torch.no_grad()
    def enhance(self, img: Image.Image) -> Image.Image:
        """
        Runs PyTorch model inference on PIL Image.
        Uses configurable tiles (default 384px) with 64px feathered overlap.
        Only tiles and a row band become float32; full images remain uint8.
        """
        oriented = ImageOps.exif_transpose(img)
        try:
            if oriented.mode == "RGB":
                return run_tiled_image(self._predict_np, oriented, tile=self.tile_size)
            with oriented.convert("RGB") as rgb:
                return run_tiled_image(self._predict_np, rgb, tile=self.tile_size)
        finally:
            oriented.close()
