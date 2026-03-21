"""Free PyTorch memory and test if CN speeds up."""
import os, sys, time, gc
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

import torch
from PIL import Image
import numpy as np
import subprocess

def gpu_mem():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
                       '--format=csv,nounits,noheader'], capture_output=True, text=True)
    used, total, free = [int(x.strip()) for x in r.stdout.strip().split(',')]
    return used, total, free

print("=== Memory Free + CN Speed Test ===", flush=True)

from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sdxl-turbo",
    t_index_list=[1], width=512, height=512,
    mode="img2img", frame_buffer_size=1,
    use_denoising_batch=True, use_lcm_lora=False, use_tiny_vae=True,
    acceleration="tensorrt", cfg_type="none", do_add_noise=True,
    warmup=5, seed=42, use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=True,
    controlnet_config=[{
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "conditioning_scale": 0.5, "preprocessor": "canny",
        "preprocessor_params": {"low_threshold": 100, "high_threshold": 200},
        "enabled": True,
    }],
    use_ipadapter=False,
)
wrapper.prepare(prompt="cyberpunk city neon", num_inference_steps=50, guidance_scale=1.0)

used, total, free = gpu_mem()
pt = torch.cuda.memory_allocated() / 1024**2
print(f"\n[Before cleanup] nvidia-smi: used={used}MB free={free}MB  PyTorch={pt:.0f}MB", flush=True)

# Audit what's on GPU
stream = wrapper.stream
print(f"\n[Audit] What's on GPU:", flush=True)
for name in ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']:
    obj = getattr(stream, name, None)
    if obj is None:
        print(f"  {name}: None", flush=True)
    elif hasattr(obj, 'device'):
        print(f"  {name}: device={obj.device} type={type(obj).__name__}", flush=True)
    elif hasattr(obj, 'parameters'):
        try:
            p = next(obj.parameters())
            print(f"  {name}: device={p.device} type={type(obj).__name__}", flush=True)
        except StopIteration:
            print(f"  {name}: no params, type={type(obj).__name__}", flush=True)
    else:
        print(f"  {name}: type={type(obj).__name__}", flush=True)

# Check ControlNet
cn_module = stream._controlnet_module
for i, cn in enumerate(cn_module.controlnets):
    if cn is not None and hasattr(cn, 'parameters'):
        try:
            p = next(cn.parameters())
            print(f"  controlnet[{i}]: device={p.device}", flush=True)
        except StopIteration:
            print(f"  controlnet[{i}]: no params", flush=True)
    else:
        print(f"  controlnet[{i}]: type={type(cn).__name__ if cn else 'None'}", flush=True)

# Try to free text encoders (they're only needed for prompt encoding, which is done)
print(f"\n[Freeing] text encoders...", flush=True)
if hasattr(stream, 'text_encoder') and stream.text_encoder is not None:
    if hasattr(stream.text_encoder, 'to'):
        stream.text_encoder.to('cpu')
        print(f"  text_encoder moved to CPU", flush=True)
if hasattr(stream, 'text_encoder_2') and stream.text_encoder_2 is not None:
    if hasattr(stream.text_encoder_2, 'to'):
        stream.text_encoder_2.to('cpu')
        print(f"  text_encoder_2 moved to CPU", flush=True)

gc.collect()
torch.cuda.empty_cache()

used, total, free = gpu_mem()
pt = torch.cuda.memory_allocated() / 1024**2
print(f"\n[After text encoder cleanup] nvidia-smi: used={used}MB free={free}MB  PyTorch={pt:.0f}MB", flush=True)

# Test CN engine speed
cn_engine = wrapper.stream.controlnet_engines[0]
sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
timestep = torch.tensor([999.0], dtype=torch.float32, device="cuda")
ehs = torch.randn(1, 77, 2048, dtype=torch.float16, device="cuda")
cc = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")
te = torch.randn(1, 1280, dtype=torch.float16, device="cuda")
ti = torch.randn(1, 6, dtype=torch.float16, device="cuda")

cn_engine._cn_call_count = 0
print(f"\n[Test] CN engine after freeing memory...", flush=True)
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = cn_engine(sample=sample, timestep=timestep, encoder_hidden_states=ehs,
                          controlnet_cond=cc, conditioning_scale=0.5, text_embeds=te, time_ids=ti)
    torch.cuda.synchronize()
    print(f"  call {i}: {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)

# Now try full pipeline
print(f"\n[Test] Full pipeline after cleanup...", flush=True)
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
cn_engine._cn_call_count = 0
for i in range(5):
    wrapper.update_control_image(0, test_img)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  frame {i}: {dt*1000:.0f}ms ({1/dt:.1f} FPS)", flush=True)

print("\nDone.", flush=True)
