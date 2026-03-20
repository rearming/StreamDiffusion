"""
Test StreamDiffusion wrapper with TensorRT on RTX 5080.
Both fixes applied:
1. utilities.py - reduced WORKSPACE/TACTIC_DRAM for Blackwell LLVM OOM
2. wrapper.py - removed aggressive CUDA context reset causing segfaults
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"GPU mem free: {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB", flush=True)

import tensorrt as trt
print(f"TensorRT: {trt.__version__}", flush=True)
print(flush=True)

from streamdiffusion import StreamDiffusionWrapper

print("=== Creating StreamDiffusionWrapper with TensorRT ===", flush=True)
t0 = time.time()

stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    acceleration="tensorrt",
    mode="txt2img",
    use_denoising_batch=True,
    cfg_type="none",
    seed=42,
    use_lcm_lora=False,  # sd-turbo doesn't need LCM LoRA
    use_tiny_vae=True,
    engine_dir="engines/test",
)

t1 = time.time()
print(f"\nWrapper created in {t1-t0:.1f}s", flush=True)

stream.prepare(
    prompt="a beautiful mountain landscape",
    num_inference_steps=50,
)
print("Prepared!", flush=True)

# Warmup
print("Warming up...", flush=True)
for _ in range(stream.batch_size - 1):
    stream()

# Benchmark
print("Benchmarking (20 frames)...", flush=True)
times = []
for i in range(20):
    t0 = time.time()
    output_image = stream()
    t1 = time.time()
    times.append(t1 - t0)

avg = sum(times) / len(times)
fps = 1.0 / avg
print(f"Average: {avg*1000:.1f}ms per frame = {fps:.1f} FPS", flush=True)
print(f"Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms", flush=True)
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
output_image.save(os.path.join(outdir, "test_trt.png"))
print("Saved! SUCCESS!", flush=True)
