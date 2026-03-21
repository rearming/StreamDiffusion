"""Minimal TensorRT test — no Gradio, no video, just load + single inference."""
import os, sys, time
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from PIL import Image
import numpy as np

print("=== TensorRT Pipeline Test ===", flush=True)

print("[1] Importing StreamDiffusionWrapper...", flush=True)
from streamdiffusion import StreamDiffusionWrapper

print("[2] Creating wrapper (sdxl-turbo, tensorrt, no controlnet, no ipadapter)...", flush=True)
wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sdxl-turbo",
    t_index_list=[1],
    width=512, height=512,
    mode="img2img",
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=False,
    use_tiny_vae=True,
    acceleration="tensorrt",
    cfg_type="none",
    do_add_noise=True,
    warmup=5,
    seed=42,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=False,
    use_ipadapter=False,
)

print("[3] Preparing...", flush=True)
wrapper.prepare(prompt="cyberpunk city neon", num_inference_steps=50, guidance_scale=1.0)

print("[4] Creating test image...", flush=True)
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

print("[5] Warmup (5 frames)...", flush=True)
for i in range(5):
    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    dt = time.perf_counter() - t0
    print(f"  warmup {i}: {dt*1000:.0f}ms", flush=True)

print("[6] Benchmark (20 frames)...", flush=True)
times = []
for i in range(20):
    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  frame {i}: {dt*1000:.0f}ms ({1/dt:.1f} FPS)", flush=True)

avg = sum(times) / len(times)
print(f"\n=== RESULT: avg={avg*1000:.0f}ms ({1/avg:.1f} FPS) ===", flush=True)
