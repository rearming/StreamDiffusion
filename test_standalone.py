"""
Standalone StreamDiffusion test - no TD, no OSC, no SharedMemory.
Generates frames, changes prompt mid-stream, saves as video.
"""
import os
import sys
import time

os.environ['HF_HOME'] = 'X:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, 'X:/td/StreamDiffusion/src')
sys.path.insert(0, 'X:/td/StreamDiffusion/StreamDiffusionTD')

import torch
import numpy as np
from PIL import Image

print("Loading model...")
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[1],
    width=512,
    height=512,
    mode="txt2img",
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=True,
    use_tiny_vae=True,
    acceleration="none",
    cfg_type="none",
    do_add_noise=True,
    warmup=5,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=False,
    use_ipadapter=False,
)

wrapper.prepare(
    prompt="psychedelic neon fractal, vibrant colors",
    num_inference_steps=50,
    guidance_scale=1.0,
)

print("Model loaded! Generating frames...\n")

frames = []
prompts_schedule = [
    (0,  "psychedelic neon fractal, vibrant colors"),
    (30, "underwater coral reef, bioluminescent creatures"),
    (60, "cyberpunk city, neon rain at night"),
    (90, "abstract liquid metal, chrome reflections"),
]

prompt_idx = 0
total_frames = 120

for i in range(total_frames):
    # Change prompt on schedule
    if prompt_idx < len(prompts_schedule) - 1 and i >= prompts_schedule[prompt_idx + 1][0]:
        prompt_idx += 1
        new_prompt = prompts_schedule[prompt_idx][1]
        print(f"\n>>> Frame {i}: Changing prompt to: '{new_prompt}'")
        wrapper.stream._param_updater.update_stream_params(
            prompt_list=[(new_prompt, 1.0)]
        )

    # Generate
    t0 = time.time()
    image = wrapper.txt2img()
    dt = time.time() - t0

    # Collect frame
    if isinstance(image, Image.Image):
        frames.append(image.copy())
    elif isinstance(image, list):
        frames.append(image[0].copy())

    fps = 1.0 / dt if dt > 0 else 0
    print(f"  Frame {i:3d}/{total_frames} | {fps:.1f} fps | prompt: {prompts_schedule[prompt_idx][1][:40]}...", end='\r')

print(f"\n\nGenerated {len(frames)} frames. Saving...")

# Save as individual PNGs for first/last of each prompt
os.makedirs("X:/td/StreamDiffusion/test_output", exist_ok=True)
for idx in [0, 29, 30, 59, 60, 89, 90, 119]:
    if idx < len(frames):
        frames[idx].save(f"X:/td/StreamDiffusion/test_output/frame_{idx:04d}.png")
        print(f"  Saved frame_{idx:04d}.png")

# Save as GIF
print("Saving animated GIF (may take a moment)...")
frames[0].save(
    "X:/td/StreamDiffusion/test_output/output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=33,  # ~30fps
    loop=0
)
print(f"\nDone! Check X:/td/StreamDiffusion/test_output/")
print(f"  - output.gif (animated)")
print(f"  - frame_XXXX.png (key frames)")
