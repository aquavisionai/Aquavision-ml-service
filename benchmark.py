"""CPU/device benchmarks with actual process high-water RSS (Linux/macOS).

Run each resolution in a fresh process; high-water RSS is cumulative within a process.
Synthetic inputs measure resource usage, not underwater enhancement quality.
"""
import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from inference import AquaVisionModel, CHECKPOINT_PATH


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_benchmark(args):
    initial_peak = peak_rss_mib()
    started = time.perf_counter()
    model = AquaVisionModel(args.checkpoint, device=args.device, tile_size=args.tile)
    synchronize(model.device)
    load_seconds = time.perf_counter() - started
    load_peak = peak_rss_mib()
    if args.image:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    else:
        image = Image.new("RGB", (args.width, args.height), color=(0, 105, 148))
    times = []
    with image:
        for _ in range(args.runs + 1):
            synchronize(model.device)
            started = time.perf_counter()
            with model.enhance(image) as enhanced:
                if enhanced.size != image.size:
                    raise AssertionError("Output dimensions differ from RGB benchmark input.")
                synchronize(model.device)
            times.append(time.perf_counter() - started)
    result = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
              "device": model.device, "threads": torch.get_num_threads(), "tile_size": model.tile_size,
              "model_version": model.model_version,
              "parameters": sum(p.numel() for p in model.model.parameters()),
              "checkpoint_bytes": args.checkpoint.stat().st_size, "input_source": "file" if args.image else "synthetic",
              "width": image.width, "height": image.height, "load_seconds": round(load_seconds, 4),
              "cold_seconds": round(times[0], 4), "warm_runs": args.runs,
              "warm_mean_seconds": round(statistics.mean(times[1:]), 4) if args.runs else None,
              "initial_peak_rss_mib": round(initial_peak, 2), "loaded_peak_rss_mib": round(load_peak, 2),
              "peak_rss_mib": round(peak_rss_mib(), 2)}
    if model.device == "cuda":
        result["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--runs", type=int, default=3, help="Warm runs after one cold inference")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--tile", type=int, default=384, help="Tile size; 192 is the low-memory profile")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--image", type=Path, help="Optional representative real image")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.runs < 0:
        parser.error("Dimensions must be positive and runs must be nonnegative.")
    run_benchmark(args)
