import os
import time
import torch
from diffusers import AutoPipelineForText2Image

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Step 1: Direct diffusers test
print("=== Test 1: Direct diffusers sd-turbo ===")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
print("Model loaded on GPU")

t0 = time.time()
image = pipe("a beautiful mountain landscape with a lake", num_inference_steps=1, guidance_scale=0.0).images[0]
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s")

outdir = os.path.join(os.path.dirname(__file__), "images", "outputs")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "test_direct.png")
image.save(outpath)
print(f"Saved to {outpath}")

# Cleanup
del pipe
torch.cuda.empty_cache()

# Step 2: StreamDiffusion wrapper test (no TensorRT)
print()
print("=== Test 2: StreamDiffusion wrapper (acceleration=none) ===")
from streamdiffusion import StreamDiffusionWrapper

stream = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[0, 16, 32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    acceleration="none",
    mode="txt2img",
    use_denoising_batch=False,
    cfg_type="none",
    seed=42,
)

stream.prepare(
    prompt="a colorful abstract painting",
    num_inference_steps=50,
)

print("Warming up...")
for _ in range(stream.batch_size - 1):
    stream()

t0 = time.time()
output_image = stream()
t1 = time.time()
print(f"Generated in {t1-t0:.2f}s")

outpath2 = os.path.join(outdir, "test_stream.png")
output_image.save(outpath2)
print(f"Saved to {outpath2}")
print()
print("=== All tests passed! ===")
