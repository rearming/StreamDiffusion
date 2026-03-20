import os
import time
import gc
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

from diffusers import StableDiffusionPipeline

model_id = "stabilityai/sd-turbo"

# Load the full pipeline at once using from_pretrained
print("Loading full pipeline...", flush=True)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
)
print("Pipeline loaded on CPU", flush=True)

print("Moving to GPU...", flush=True)
pipe = pipe.to("cuda")
print(f"Pipeline on GPU, mem: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print("Generating...", flush=True)
t0 = time.time()
image = pipe("a beautiful mountain landscape with snow", num_inference_steps=1, guidance_scale=0.0).images[0]
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s", flush=True)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
image.save(os.path.join(outdir, "test_full3.png"))
print("Saved! SUCCESS!", flush=True)
