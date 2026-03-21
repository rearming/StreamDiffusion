"""Diagnose CN slowdown in pipeline context — check memory + stream issues."""
import os, sys, time
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

import torch
from polygraphy import cuda

print("=== CN Pipeline Diagnosis ===", flush=True)

# Load the full pipeline
from streamdiffusion import StreamDiffusionWrapper
from PIL import Image
import numpy as np

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
wrapper.prepare(prompt="cyberpunk city neon", num_inference_steps=50, guidance_scale=1.0)

print(f"\n[VRAM] allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB  cached={torch.cuda.memory_reserved()/1024**3:.2f}GB", flush=True)

# Get the CN engine
cn_engines = wrapper.stream.controlnet_engines
cn_engine = cn_engines[0]

# Create test inputs matching pipeline shapes
sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
timestep = torch.tensor([999.0], dtype=torch.float32, device="cuda")
encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16, device="cuda")
controlnet_cond = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")
text_embeds = torch.randn(1, 1280, dtype=torch.float16, device="cuda")
time_ids = torch.randn(1, 6, dtype=torch.float16, device="cuda")

# Test 1: CN engine directly in pipeline context (all engines loaded)
print("\n[Test 1] CN engine direct call (all engines loaded)...", flush=True)
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    down, mid = cn_engine(
        sample=sample, timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_cond,
        conditioning_scale=0.5,
        text_embeds=text_embeds, time_ids=time_ids,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  call {i}: {dt*1000:.1f}ms", flush=True)

# Test 2: Full frame to see where time goes - with sync between each step
print("\n[Test 2] Full frame with per-step sync...", flush=True)
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# Warmup
for _ in range(3):
    wrapper.update_control_image(0, test_img)
    wrapper.img2img(test_img)

# Now one frame with detailed sync at every step
print("  Running one frame with per-stage sync...", flush=True)
wrapper.update_control_image(0, test_img)
torch.cuda.synchronize()

t_start = time.perf_counter()

# Step 1: preprocess
img_tensor = wrapper.preprocess_image(test_img)
torch.cuda.synchronize()
t_preprocess = time.perf_counter()

# Step 2: stream.__call__ internals - encode
x = wrapper.stream.image_processor.preprocess(img_tensor, wrapper.stream.height, wrapper.stream.width).to(
    device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t_img_preproc = time.perf_counter()

x_t_latent = wrapper.stream.encode_image(x)
torch.cuda.synchronize()
t_encode = time.perf_counter()

# Step 3: predict_x0_batch - this is where CN + UNet happen
# We need to set up the buffer first
t_list = wrapper.stream.sub_timesteps_tensor
x_t = x_t_latent  # batch=1, no denoising batch concat needed with steps_num=1

# Step 3a: CN hook manually
cn_module = wrapper.stream._controlnet_module
from streamdiffusion.hooks import StepCtx
sdxl_cond = None
if hasattr(wrapper.stream, 'add_text_embeds') and wrapper.stream.add_text_embeds is not None:
    sdxl_cond = {
        'text_embeds': wrapper.stream.add_text_embeds,
        'time_ids': wrapper.stream.add_time_ids,
    }
ctx = StepCtx(
    x_t_latent=x_t,
    t_list=t_list,
    step_index=None,
    guidance_mode="none",
    sdxl_cond=sdxl_cond,
)

torch.cuda.synchronize()
t_before_cn = time.perf_counter()

hook = wrapper.stream.unet_hooks[0]  # CN hook
delta = hook(ctx)
torch.cuda.synchronize()
t_after_cn = time.perf_counter()

# Step 3b: UNet
print(f"  Building UNet kwargs...", flush=True)
unet_kwargs = {
    'sample': x_t,
    'timestep': t_list,
    'encoder_hidden_states': wrapper.stream.prompt_embeds,
}
if wrapper.stream.is_sdxl and hasattr(wrapper.stream, 'add_text_embeds'):
    batch_size = x_t.shape[0]
    cached = wrapper.stream._get_cached_sdxl_conditioning(batch_size, wrapper.stream.cfg_type, wrapper.stream.guidance_scale)
    if cached:
        unet_kwargs.update(cached)
    else:
        cond = wrapper.stream._build_sdxl_conditioning(batch_size)
        unet_kwargs.update(cond)

extra = {}
if delta.down_block_additional_residuals is not None:
    extra['down_block_additional_residuals'] = delta.down_block_additional_residuals
if delta.mid_block_additional_residual is not None:
    extra['mid_block_additional_residual'] = delta.mid_block_additional_residual

torch.cuda.synchronize()
t_before_unet = time.perf_counter()
model_pred = wrapper.stream.unet(
    unet_kwargs['sample'], unet_kwargs['timestep'], unet_kwargs['encoder_hidden_states'],
    **extra,
    **(unet_kwargs.get('added_cond_kwargs', {}))
)[0]
torch.cuda.synchronize()
t_after_unet = time.perf_counter()

# Step 4: decode
x_0_pred = wrapper.stream.scheduler_step_batch(model_pred, x_t, wrapper.stream.sub_timesteps_tensor)
torch.cuda.synchronize()
t_sched = time.perf_counter()

x_output = wrapper.stream.decode_image(x_0_pred)
torch.cuda.synchronize()
t_decode = time.perf_counter()

print(f"\n=== BREAKDOWN (with full sync) ===")
print(f"  preprocess:     {(t_preprocess-t_start)*1000:.1f}ms")
print(f"  img_preproc:    {(t_img_preproc-t_preprocess)*1000:.1f}ms")
print(f"  VAE encode:     {(t_encode-t_img_preproc)*1000:.1f}ms")
print(f"  CN hook:        {(t_after_cn-t_before_cn)*1000:.1f}ms")
print(f"  UNet:           {(t_after_unet-t_before_unet)*1000:.1f}ms")
print(f"  Scheduler:      {(t_sched-t_after_unet)*1000:.1f}ms")
print(f"  VAE decode:     {(t_decode-t_sched)*1000:.1f}ms")
print(f"  TOTAL:          {(t_decode-t_start)*1000:.1f}ms")
