"""
hd_pipeline.py — torch-free image pipeline helpers (NumPy + Pillow only).

Why this file exists
--------------------
The AquaVision U-Net is a *same-resolution colour/contrast* model (input HxW -> output HxW).
Running it on a full-resolution photo has three problems:
  1. it never increases resolution, so a small upload can never become "HD";
  2. it was trained at a much smaller scale, so at full resolution it lifts shadows (milky look)
     and amplifies grain instead of adding detail;
  3. memory grows with the pixel count (a 12 MP photo can exhaust RAM).

The pipeline below fixes that:
  * the U-Net runs on a reduced copy of the image (``work_long_side``)          -> colour / haze fix
  * only its *correction* (enhanced - input) is upsampled and added to the
    full-resolution original                                                      -> keeps original sharpness
  * optionally a super-resolution model upscales small images to HD / 4K        -> real HD output
  * optionally a light unsharp mask                                             -> crisper finish

Everything here takes a ``predict_fn`` (HxWx3 float32 in [0,1] -> HxWx3 float32), so the logic is
unit-testable without PyTorch (see tests/test_service.py).

This HD/SR pipeline is experimental. The HTTP API uses run_tiled_image instead;
the float-array helpers below still scale memory with the full image size.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from PIL import Image

PredictFn = Callable[[np.ndarray], np.ndarray]

_LANCZOS = Image.Resampling.LANCZOS
_BICUBIC = Image.Resampling.BICUBIC
_BOX = Image.Resampling.BOX


# --------------------------------------------------------------------------- basic helpers
def resize_f32(arr: np.ndarray, size_wh: Tuple[int, int], resample) -> np.ndarray:
    """Resize an HxWxC float32 array (values may leave 0..1, e.g. a delta) channel by channel."""
    w, h = int(size_wh[0]), int(size_wh[1])
    chans = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(arr[..., c], dtype=np.float32)).resize((w, h), resample),
            dtype=np.float32,
        )
        for c in range(arr.shape[-1])
    ]
    return np.stack(chans, axis=-1)


def fit_long_side(shape_hw: Tuple[int, int], long_side: int) -> Tuple[int, int]:
    """Return (width, height) whose longer side equals ``long_side`` (aspect ratio preserved)."""
    h, w = shape_hw
    s = long_side / float(max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def to_uint8(rgb01: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint8 with proper rounding (torchvision's ToPILImage truncates, biasing dark)."""
    return np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- tiled inference
def _starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride))
    s.append(length - tile)
    return s


def _window(n: int, ramp: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    ramp = min(ramp, n // 2)
    for i in range(ramp):
        v = (i + 1) / (ramp + 1)
        w[i] = min(w[i], v)
        w[n - 1 - i] = min(w[n - 1 - i], v)
    return w


def run_tiled(predict_fn: PredictFn, arr: np.ndarray, tile: int = 512, overlap: int = 32, scale: int = 1) -> np.ndarray:
    """Run ``predict_fn`` over overlapping tiles and blend with a feathered window (no visible seams).

    ``scale`` is the output/input size ratio of ``predict_fn`` (1 for the U-Net, 2 or 4 for super-resolution).
    """
    H, W, C = arr.shape
    if H <= tile and W <= tile:
        return predict_fn(arr)

    stride = max(1, tile - overlap)
    th, tw = min(tile, H), min(tile, W)
    out = np.zeros((H * scale, W * scale, C), dtype=np.float32)
    wsum = np.zeros((H * scale, W * scale, 1), dtype=np.float32)
    wy = _window(th * scale, overlap * scale)
    wx = _window(tw * scale, overlap * scale)
    w2 = (wy[:, None] * wx[None, :])[..., None]

    for y in _starts(H, tile, stride):
        for x in _starts(W, tile, stride):
            pred = predict_fn(np.ascontiguousarray(arr[y:y + th, x:x + tw]))
            ys, xs = y * scale, x * scale
            out[ys:ys + th * scale, xs:xs + tw * scale] += pred * w2
            wsum[ys:ys + th * scale, xs:xs + tw * scale] += w2
    return out / np.maximum(wsum, 1e-6)


def run_tiled_image(predict_fn: PredictFn, image: Image.Image, tile: int = 384, overlap: int = 64) -> Image.Image:
    """Same feathered predictions as run_tiled, with only one row of float tiles.

    Input/output stay as uint8 PIL images. Float buffers use O(tile * width)
    memory instead of O(height * width); only completed rows are encoded.
    """
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Require tile > 0 and 0 <= overlap < tile.")
    width, height = image.size
    if width <= tile and height <= tile:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(to_uint8(predict_fn(rgb)))

    th, tw = min(tile, height), min(tile, width)
    ys = _starts(height, tile, tile - overlap)
    xs = _starts(width, tile, tile - overlap)
    weights = (_window(th, overlap)[:, None] * _window(tw, overlap)[None, :])[..., None]
    band = np.zeros((th, width, 3), dtype=np.float32)
    weight_sum = np.zeros((th, width, 1), dtype=np.float32)
    output = Image.new("RGB", (width, height))

    for index, y in enumerate(ys):
        for x in xs:
            with image.crop((x, y, x + tw, y + th)) as crop:
                rgb = np.asarray(crop, dtype=np.float32) / 255.0
            prediction = predict_fn(rgb)
            if prediction.shape != rgb.shape:
                raise ValueError("Predictor must preserve RGB tile dimensions.")
            band[:, x:x + tw] += prediction * weights
            weight_sum[:, x:x + tw] += weights
        # Subsequent tiles cannot contribute to these completed rows.
        rows = ys[index + 1] - y if index + 1 < len(ys) else height - y
        band[:rows] /= np.maximum(weight_sum[:rows], 1e-6)
        with Image.fromarray(to_uint8(band[:rows])) as completed:
            output.paste(completed, (0, y))
        if rows < th:
            band[:th - rows] = band[rows:]
            weight_sum[:th - rows] = weight_sum[rows:]
        band[th - rows:] = 0
        weight_sum[th - rows:] = 0
    return output


# --------------------------------------------------------------------------- core enhancement
def detail_preserving_enhance(
    rgb01: np.ndarray,
    predict_fn: PredictFn,
    work_long_side: int = 768,
    native_tile: int = 512,
    native_overlap: int = 64,
) -> np.ndarray:
    """Apply the U-Net's colour/haze correction without destroying the original's detail.

    work_long_side > 0 : run the model on a reduced copy, upsample (enhanced - input), add to the full-res original.
    work_long_side <= 0: run the model at native resolution (tiled, memory safe) — the old behaviour.
    Images already smaller than ``work_long_side`` are simply run as-is.
    """
    H, W, _ = rgb01.shape
    if work_long_side <= 0:
        return np.clip(run_tiled(predict_fn, rgb01, native_tile, native_overlap), 0, 1)

    scale = min(1.0, work_long_side / float(max(H, W)))
    if scale >= 1.0:
        return np.clip(predict_fn(rgb01), 0, 1)

    w, h = max(8, round(W * scale)), max(8, round(H * scale))
    small = np.clip(resize_f32(rgb01, (w, h), _BOX if scale <= 0.5 else _LANCZOS), 0, 1)
    enhanced_small = predict_fn(small)
    delta = resize_f32(enhanced_small - small, (W, H), _BICUBIC)
    return np.clip(rgb01 + delta, 0, 1)


# --------------------------------------------------------------------------- post-processing
def _gaussian_blur_2d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i in range(len(k)):
            acc += k[i] * (p[i:i + n, :] if axis == 0 else p[:, i:i + n])
        out = acc
    return out


def sharpen_luma(rgb01: np.ndarray, amount: float = 0.3, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask on luminance only (adds the same detail to R, G, B so hue does not shift)."""
    if amount <= 0:
        return rgb01
    luma = rgb01 @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    detail = luma - _gaussian_blur_2d(luma, sigma)
    return np.clip(rgb01 + amount * detail[..., None], 0, 1)


# --------------------------------------------------------------------------- dehazing (dark channel prior)
def _min_filter_2d(a: np.ndarray, patch: int) -> np.ndarray:
    """Separable minimum filter (patch x patch), edge padded."""
    r = patch // 2
    out = a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = p[0:n, :] if axis == 0 else p[:, 0:n]
        for i in range(1, patch):
            acc = np.minimum(acc, p[i:i + n, :] if axis == 0 else p[:, i:i + n])
        out = acc
    return out


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="edge")
    c = p.cumsum(0).cumsum(1)
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)).astype(np.float32)


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    mg, ms = _box_mean(guide, r), _box_mean(src, r)
    cov = _box_mean(guide * src, r) - mg * ms
    var = _box_mean(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return _box_mean(a, r) * guide + _box_mean(b, r)


def dehaze_transfer(
    rgb01: np.ndarray,
    strength: float = 0.6,
    work_long_side: int = 768,
    patch: int = 15,
    t_min: float = 0.25,
) -> np.ndarray:
    """Remove the veil of haze / backscatter (dark channel prior).

    The veil colour and transmission map are estimated on a reduced copy (fast, resolution independent) and then
    applied to the full-resolution image, so it stays cheap and sharp on big photos.
    strength 0 = off, 0.5-0.7 = natural, 0.8+ = strong (can amplify noise).
    """
    if strength <= 0:
        return rgb01
    H, W, _ = rgb01.shape
    scale = min(1.0, max(work_long_side, 64) / float(max(H, W)))
    if scale < 1.0:
        w, h = max(16, round(W * scale)), max(16, round(H * scale))
        small = np.clip(resize_f32(rgb01, (w, h), _BOX if scale <= 0.5 else _LANCZOS), 0, 1)
    else:
        small = rgb01

    dark = _min_filter_2d(small.min(-1), patch)
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    veil = np.maximum(small.reshape(-1, 3)[idx].mean(0), 1e-3)                       # veil colour A
    t = 1.0 - strength * _min_filter_2d((small / veil).min(-1), patch)
    t = np.clip(_guided_filter(small.mean(-1), t, max(4, round(0.07 * min(small.shape[:2]))), 1e-3), t_min, 1.0)

    if scale < 1.0:
        t = np.clip(resize_f32(t[..., None], (W, H), _BICUBIC)[..., 0], t_min, 1.0)
    return np.clip((rgb01 - veil) / t[..., None] + veil, 0, 1)


# --------------------------------------------------------------------------- full pipeline
def hd_enhance(
    rgb01: np.ndarray,
    predict_fn: PredictFn,
    *,
    work_long_side: int = 768,
    dehaze_strength: float = 0.0,
    sr_predict_fn: PredictFn | None = None,
    sr_scale: int = 4,
    sr_max_input_long_side: int = 1280,
    sr_target_long_side: int = 1920,
    keep_original_size: bool = True,
    sharpen_amount: float = 0.0,
) -> np.ndarray:
    """U-Net colour correction -> dehaze -> (optional) super-resolution -> (optional) sharpening.

    keep_original_size=True : super-resolution is used as a "clean-up" (upscale x4, then scale back down), so the
                              picture gets clearer and less noisy but keeps the SAME width and height.
    keep_original_size=False: the result is scaled to ``sr_target_long_side`` (true HD / 4K output).
    Returns float [0,1].
    """
    H, W, _ = rgb01.shape
    out = detail_preserving_enhance(rgb01, predict_fn, work_long_side)
    out = dehaze_transfer(out, dehaze_strength, work_long_side if work_long_side > 0 else 768)

    if sr_predict_fn is not None and max(H, W) < sr_max_input_long_side:
        out = np.clip(run_tiled(sr_predict_fn, out, tile=256, overlap=16, scale=sr_scale), 0, 1)
        if keep_original_size:
            out = np.clip(resize_f32(out, (W, H), _LANCZOS), 0, 1)
        elif sr_target_long_side and max(out.shape[:2]) != sr_target_long_side:
            downscaling = max(out.shape[:2]) > sr_target_long_side
            out = np.clip(
                resize_f32(out, fit_long_side(out.shape[:2], sr_target_long_side), _LANCZOS if downscaling else _BICUBIC),
                0, 1,
            )

    return sharpen_luma(out, sharpen_amount) if sharpen_amount > 0 else out
