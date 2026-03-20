"""
Test StreamDiffusion core (pipeline.py) directly, bypassing the wrapper's
CUDA context reset which causes segfaults on RTX 5080.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(flush=True)

# Step 1: Load pipeline with diffusers (this works reliably)
from diffusers import StableDiffusionPipeline

print("Loading sd-turbo pipeline...", flush=True)
# Load components individually to avoid intermittent segfaults
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

model_id = "stabilityai/sd-turbo"

print("  Loading tokenizer...", flush=True)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

print("  Loading scheduler...", flush=True)
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

print("  Loading UNet (largest component first)...", flush=True)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)

print("  Loading text_encoder...", flush=True)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)

print("  Loading VAE...", flush=True)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    unet=unet, scheduler=scheduler,
    safety_checker=None, feature_extractor=None,
)
print("Pipeline assembled on CPU, moving to GPU...", flush=True)
pipe = pipe.to("cuda")
print(f"Pipeline on GPU, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

# Step 2: Create StreamDiffusion core
from streamdiffusion import StreamDiffusion

print("Creating StreamDiffusion...", flush=True)
stream = StreamDiffusion(
    pipe=pipe,
    t_index_list=[0, 16, 32, 45],
    device="cuda",
    torch_dtype=torch.float16,
    width=512,
    height=512,
    do_add_noise=True,
    use_denoising_batch=True,
    frame_buffer_size=1,
    cfg_type="none",
)
print(f"StreamDiffusion created, model type: {stream.model_type}", flush=True)

# Step 3: sd-turbo is already a turbo model, no LCM LoRA needed

# Step 4: Load tiny VAE for speed
from diffusers import AutoencoderTiny
print("Loading TinyVAE...", flush=True)
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
    device=stream.device, dtype=stream.dtype
)
print("TinyVAE loaded", flush=True)

# Step 5: Move to GPU and prepare
print("Preparing...", flush=True)
stream.prepare(
    prompt="a colorful abstract painting with geometric shapes",
    num_inference_steps=50,
    guidance_scale=0.0,
)
print(f"Prepared! GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

# Step 6: Warmup
print("Warming up...", flush=True)
for _ in range(stream.batch_size):
    stream.txt2img()
print("Warmup done", flush=True)

# Step 7: Benchmark
print("Benchmarking (10 frames)...", flush=True)
times = []
for i in range(10):
    t0 = time.time()
    image_tensor = stream.txt2img()
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1 - t0)

avg = sum(times) / len(times)
fps = 1.0 / avg
print(f"Average: {avg*1000:.1f}ms per frame = {fps:.1f} FPS", flush=True)
print(f"Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms", flush=True)

# Save last frame
from streamdiffusion.image_utils import postprocess_image
output_image = postprocess_image(image_tensor, output_type="pil")[0]
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
output_image.save(os.path.join(outdir, "test_core.png"))
print(f"Saved to images/outputs/test_core.png", flush=True)

print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
print("SUCCESS!", flush=True)
