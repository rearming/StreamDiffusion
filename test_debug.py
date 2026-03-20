import torch
import gc

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# Try loading individual components
print("\n--- Loading tokenizer ---")
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
print("OK")

print("\n--- Loading text encoder ---")
from transformers import CLIPTextModel
text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder = text_encoder.to("cuda")
print(f"OK, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("\n--- Loading VAE ---")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae", torch_dtype=torch.float16)
vae = vae.to("cuda")
print(f"OK, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("\n--- Loading UNet ---")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet", torch_dtype=torch.float16)
unet = unet.to("cuda")
print(f"OK, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("\n--- Loading scheduler ---")
from diffusers import EulerDiscreteScheduler
scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
print("OK")

print("\nAll components loaded successfully!")

# Quick inference test
print("\n--- Running inference ---")
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
)
import time
t0 = time.time()
image = pipe("mountain landscape", num_inference_steps=1, guidance_scale=0.0).images[0]
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s")

import os
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
image.save(os.path.join(outdir, "test_debug.png"))
print("Saved! All good.")
