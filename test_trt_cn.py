"""Test TRT pipeline WITH ControlNet — measure where time goes."""
import os, sys, time
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from PIL import Image
import numpy as np
import torch

print("=== TRT + ControlNet Pipeline Test ===", flush=True)
from streamdiffusion import StreamDiffusionWrapper

print("[1] Loading pipeline...", flush=True)
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
    use_controlnet=True,
    controlnet_config=[{
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "conditioning_scale": 0.5,
        "preprocessor": "canny",
        "preprocessor_params": {"low_threshold": 100, "high_threshold": 200},
        "enabled": True,
    }],
    use_ipadapter=False,
)

print("[2] Preparing...", flush=True)
wrapper.prepare(prompt="cyberpunk city neon", num_inference_steps=50, guidance_scale=1.0)

test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# Check what type of CN is being used
cn_module = wrapper.stream._controlnet_module
cn_engines = getattr(wrapper.stream, 'controlnet_engines', [])
print(f"[3] CN module has {len(cn_module.controlnets)} controlnets", flush=True)
print(f"    CN engines on stream: {len(cn_engines)}", flush=True)
for eng in cn_engines:
    print(f"    Engine: {type(eng).__name__} model_id={getattr(eng, 'model_id', '?')}", flush=True)

print("[4] Warmup (3 frames)...", flush=True)
for i in range(3):
    wrapper.update_control_image(0, test_img)
    torch.cuda.synchronize()  # ensure CN preprocessing is done
    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  warmup {i}: {dt*1000:.0f}ms", flush=True)

print("[5] Benchmark (10 frames)...", flush=True)
# Add call counter to CN engine
cn_eng = wrapper.stream.controlnet_engines[0]
cn_eng._bench_calls = 0
_orig_call = cn_eng.__class__.__call__
def _counted_call(self, *a, **kw):
    self._bench_calls += 1
    return _orig_call(self, *a, **kw)
cn_eng.__class__.__call__ = _counted_call

times = []
for i in range(10):
    cn_eng._bench_calls = 0
    wrapper.update_control_image(0, test_img)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  frame {i}: {dt*1000:.0f}ms ({1/dt:.1f} FPS) cn_calls={cn_eng._bench_calls}", flush=True)

avg = sum(times) / len(times)
print(f"\n=== RESULT: avg={avg*1000:.0f}ms ({1/avg:.1f} FPS) ===", flush=True)
