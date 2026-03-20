import sys
import os
import time
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionPipeline

model_id = "stabilityai/sd-turbo"
dtype = torch.float16

print("Loading tokenizer...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
print("OK")

print("Loading text encoder...")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder = text_encoder.to("cuda")
print(f"OK, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
vae = vae.to("cuda")
print(f"OK, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
unet = unet.to("cuda")
print(f"OK, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("Loading scheduler...")
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
print("OK")

print("Building pipeline...")
pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
    unet=unet, scheduler=scheduler,
    safety_checker=None, feature_extractor=None,
)
print("Pipeline ready!")

print("Generating image...")
t0 = time.time()
image = pipe("a beautiful mountain landscape", num_inference_steps=1, guidance_scale=0.0).images[0]
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s")

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "test_full.png")
image.save(outpath)
print(f"Saved to {outpath}")
print("SUCCESS!")
