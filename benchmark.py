import os
import time
import tracemalloc
from pathlib import Path

import psutil
import torch
from PIL import Image

try:
    from inference import AquaVisionModel
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from inference import AquaVisionModel

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    print("Starting AquaVision Local Benchmark...")
    print("-" * 40)
    
    # Measure Checkpoint Size
    checkpoint_path = Path(__file__).parent / "best.pt"
    if checkpoint_path.exists():
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Checkpoint size: {ckpt_size_mb:.2f} MB")
    else:
        print("Checkpoint not found!")
        return

    # Create a dummy image for testing (e.g., 800x600)
    img_w, img_h = 800, 600
    dummy_img = Image.new('RGB', (img_w, img_h), color=(0, 105, 148))
    print(f"Input resolution: {img_w}x{img_h}")

    # Measure RAM before loading
    mem_before = get_memory_mb()
    print(f"RAM before loading model: {mem_before:.2f} MB")

    tracemalloc.start()
    
    # Load Model
    start_load = time.time()
    model = AquaVisionModel(checkpoint_path=checkpoint_path)
    load_time = time.time() - start_load
    
    mem_after = get_memory_mb()
    print(f"RAM after loading model: {mem_after:.2f} MB")
    print(f"Model load time: {load_time:.2f} seconds")
    print(f"Device: {model.device}")

    if model.device == 'cuda':
        print(f"GPU VRAM allocated: {torch.cuda.memory_allocated() / (1024*1024):.2f} MB")

    # Cold Inference
    print("Running cold inference...")
    start_cold = time.time()
    out_img = model.enhance(dummy_img)
    cold_time = time.time() - start_cold
    print(f"Cold inference time: {cold_time:.2f} seconds")
    print(f"Output resolution: {out_img.width}x{out_img.height}")

    # Warm Inference
    print("Running warm inference (average of 3 runs)...")
    warm_times = []
    for i in range(3):
        start_warm = time.time()
        _ = model.enhance(dummy_img)
        warm_times.append(time.time() - start_warm)
    
    avg_warm = sum(warm_times) / len(warm_times)
    print(f"Warm inference average time: {avg_warm:.2f} seconds")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / (1024 * 1024)
    print(f"Peak Python traced memory during run: {peak_mb:.2f} MB")
    print(f"Peak Process RAM: {get_memory_mb():.2f} MB")

    print("-" * 40)
    print("Benchmark Complete!")

if __name__ == '__main__':
    run_benchmark()
