"""
Test StreamDiffusion with TensorRT in img2img mode on RTX 5080.
This is the primary use case - real-time image-to-image transformation.
"""
import os
import time
import torch
from PIL import Image

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

from streamdiffusion import StreamDiffusionWrapper

# Create a test input image (solid color gradient)
print("Creating test input image...", flush=True)
import numpy as np
arr = np.zeros((512, 512, 3), dtype=np.uint8)
for y in range(512):
    for x in range(512):
        arr[y, x] = [int(x/2), int(y/2), 128]
input_image = Image.fromarray(arr)
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
input_image.save(os.path.join(outdir, "test_input.png"))

print("=== Creating StreamDiffusionWrapper (img2img + TensorRT) ===", flush=True)
t0 = time.time()

stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    acceleration="tensorrt",
    mode="img2img",
    use_denoising_batch=True,
    cfg_type="self",
    seed=42,
    use_lcm_lora=False,
    use_tiny_vae=True,
    engine_dir="engines/test",
)

t1 = time.time()
print(f"Wrapper created in {t1-t0:.1f}s", flush=True)

stream.prepare(
    prompt="a beautiful watercolor painting of mountains",
    negative_prompt="ugly, blurry",
    num_inference_steps=50,
    guidance_scale=1.2,
    delta=0.5,
)
print("Prepared!", flush=True)

# Preprocess input
image_tensor = stream.preprocess_image(input_image)

# Warmup
print("Warming up...", flush=True)
for _ in range(stream.batch_size - 1):
    stream(image=image_tensor)

# Benchmark
print("Benchmarking (20 frames)...", flush=True)
times = []
for i in range(20):
    t0 = time.time()
    output_image = stream(image=image_tensor)
    t1 = time.time()
    times.append(t1 - t0)

avg = sum(times) / len(times)
fps = 1.0 / avg
print(f"Average: {avg*1000:.1f}ms per frame = {fps:.1f} FPS", flush=True)
print(f"Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms", flush=True)
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

output_image.save(os.path.join(outdir, "test_trt_img2img.png"))
print("Saved! SUCCESS!", flush=True)
