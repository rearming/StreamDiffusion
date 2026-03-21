"""Standalone CN TRT engine benchmark — no pipeline, no hooks, just raw engine."""
import os, sys, time
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

import torch
from polygraphy import cuda

print("=== Standalone CN TRT Engine Benchmark ===", flush=True)

from streamdiffusion.acceleration.tensorrt.runtime_engines.controlnet_engine import ControlNetModelEngine

engine_path = "X:/td/StreamDiffusion/engines/td/cn_test/controlnet.engine"
stream = cuda.Stream()

print("[1] Loading engine...", flush=True)
engine = ControlNetModelEngine(engine_path, stream, use_cuda_graph=False, model_type="sdxl")
print("    Done.", flush=True)

# Create dummy inputs matching what pipeline sends
sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
timestep = torch.tensor([999.0], dtype=torch.float32, device="cuda")
encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16, device="cuda")
controlnet_cond = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")
text_embeds = torch.randn(1, 1280, dtype=torch.float16, device="cuda")
time_ids = torch.randn(1, 6, dtype=torch.float16, device="cuda")

print("[2] Warmup (3 calls)...", flush=True)
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = engine(
        sample=sample, timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_cond,
        conditioning_scale=0.5,
        text_embeds=text_embeds, time_ids=time_ids,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  warmup {i}: {dt*1000:.1f}ms", flush=True)

print("[3] Benchmark (10 calls, sync'd)...", flush=True)
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = engine(
        sample=sample, timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_cond,
        conditioning_scale=0.5,
        text_embeds=text_embeds, time_ids=time_ids,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    times.append(dt * 1000)
    print(f"  call {i}: {dt*1000:.1f}ms", flush=True)

avg = sum(times) / len(times)
print(f"\n=== CN ENGINE STANDALONE: avg={avg:.1f}ms ===", flush=True)
print(f"    Output: {len(down)} down blocks, mid={tuple(mid.shape)}", flush=True)
