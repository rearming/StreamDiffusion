import os
import time
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionPipeline

model_id = "stabilityai/sd-turbo"
dtype = torch.float16

# Load everything to CPU first, then move to GPU
print("Loading all components to CPU...", flush=True)

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
print("  tokenizer OK", flush=True)

text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
print("  text_encoder OK", flush=True)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
print("  vae OK", flush=True)

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
print("  unet OK", flush=True)

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
print("  scheduler OK", flush=True)

print("Moving to GPU...", flush=True)
text_encoder = text_encoder.to("cuda")
print(f"  text_encoder -> cuda, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
vae = vae.to("cuda")
print(f"  vae -> cuda, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
unet = unet.to("cuda")
print(f"  unet -> cuda, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print("Building pipeline...", flush=True)
pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    unet=unet, scheduler=scheduler,
    safety_checker=None, feature_extractor=None,
)

print("Generating...", flush=True)
t0 = time.time()
image = pipe("a beautiful mountain landscape", num_inference_steps=1, guidance_scale=0.0).images[0]
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s", flush=True)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
image.save(os.path.join(outdir, "test_full2.png"))
print("Saved! SUCCESS!", flush=True)
