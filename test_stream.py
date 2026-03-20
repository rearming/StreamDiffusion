import os
import time
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(flush=True)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamdiffusion import StreamDiffusionWrapper

print("=== StreamDiffusion txt2img (acceleration=none) ===", flush=True)

stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[0, 16, 32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    acceleration="none",
    mode="txt2img",
    use_denoising_batch=False,
    cfg_type="none",
    seed=42,
)
print("Wrapper created", flush=True)

stream.prepare(
    prompt="a colorful abstract painting with geometric shapes",
    num_inference_steps=50,
)
print("Prepared", flush=True)

# Warmup
print("Warming up...", flush=True)
for _ in range(stream.batch_size - 1):
    stream()

# Benchmark
print("Generating...", flush=True)
times = []
for i in range(10):
    t0 = time.time()
    output_image = stream()
    t1 = time.time()
    times.append(t1 - t0)

avg = sum(times) / len(times)
fps = 1.0 / avg
print(f"Average: {avg*1000:.1f}ms per frame = {fps:.1f} FPS", flush=True)
print(f"Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms", flush=True)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
output_image.save(os.path.join(outdir, "test_stream.png"))
print(f"Saved to images/outputs/test_stream.png", flush=True)
print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
print("SUCCESS!", flush=True)
