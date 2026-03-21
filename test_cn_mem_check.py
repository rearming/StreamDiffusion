"""Check GPU memory and test CN with torch cache cleared."""
import os, sys, time
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

import torch
from polygraphy import cuda
from PIL import Image
import numpy as np
import subprocess

def gpu_mem():
    """Get actual GPU memory usage from nvidia-smi"""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
                           '--format=csv,nounits,noheader'], capture_output=True, text=True)
        used, total, free = [int(x.strip()) for x in r.stdout.strip().split(',')]
        return used, total, free
    except:
        return 0, 0, 0

print("=== GPU Memory + CN Engine Analysis ===", flush=True)
used, total, free = gpu_mem()
print(f"[Before load] nvidia-smi: used={used}MB total={total}MB free={free}MB", flush=True)

from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sdxl-turbo",
    t_index_list=[1],
    width=512, height=512,
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
pt_alloc = torch.cuda.memory_allocated() / 1024**2
pt_cached = torch.cuda.memory_reserved() / 1024**2
print(f"\n[After load] nvidia-smi: used={used}MB total={total}MB free={free}MB", flush=True)
print(f"[After load] PyTorch: allocated={pt_alloc:.0f}MB cached={pt_cached:.0f}MB", flush=True)
print(f"[After load] TRT engines (non-PyTorch): ~{used - pt_cached:.0f}MB", flush=True)

# Test: clear PyTorch cache and try CN
print("\n[Test A] CN engine WITH PyTorch cache (default)...", flush=True)
cn_engine = wrapper.stream.controlnet_engines[0]
sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
timestep = torch.tensor([999.0], dtype=torch.float32, device="cuda")
ehs = torch.randn(1, 77, 2048, dtype=torch.float16, device="cuda")
cc = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")
te = torch.randn(1, 1280, dtype=torch.float16, device="cuda")
ti = torch.randn(1, 6, dtype=torch.float16, device="cuda")

# Reset CN call count
cn_engine._cn_call_count = 0

for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = cn_engine(sample=sample, timestep=timestep, encoder_hidden_states=ehs,
                          controlnet_cond=cc, conditioning_scale=0.5, text_embeds=te, time_ids=ti)
    torch.cuda.synchronize()
    print(f"  call {i}: {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)

# Now clear PyTorch cache and retry
print("\n[Test B] CN engine AFTER torch.cuda.empty_cache()...", flush=True)
torch.cuda.empty_cache()
used, total, free = gpu_mem()
pt_alloc2 = torch.cuda.memory_allocated() / 1024**2
pt_cached2 = torch.cuda.memory_reserved() / 1024**2
print(f"  nvidia-smi: used={used}MB free={free}MB", flush=True)
print(f"  PyTorch: allocated={pt_alloc2:.0f}MB cached={pt_cached2:.0f}MB", flush=True)

cn_engine._cn_call_count = 0
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = cn_engine(sample=sample, timestep=timestep, encoder_hidden_states=ehs,
                          controlnet_cond=cc, conditioning_scale=0.5, text_embeds=te, time_ids=ti)
    torch.cuda.synchronize()
    print(f"  call {i}: {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)

print("\nDone.", flush=True)
