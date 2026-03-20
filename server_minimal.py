"""
Minimal StreamDiffusion server - replaces DotSimulate's td_manager.
Direct OSC control, SharedMemory output to TouchDesigner.
"""
import os, sys, time, threading
import numpy as np
from PIL import Image
from multiprocessing import shared_memory

os.environ['HF_HOME'] = 'X:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import udp_client

# === CONFIG ===
WIDTH, HEIGHT = 512, 512
OSC_LISTEN_PORT = 8577      # We listen (TD sends here)
OSC_SEND_PORT = 8567        # We send (TD listens here)
OUTPUT_MEM_NAME = "StreamDiffusionTD_512-512_out"
INITIAL_PROMPT = "psychedelic neon fractal, vibrant colors"

# === LOAD MODEL ===
print("Loading sd-turbo...")
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[1],
    width=WIDTH, height=HEIGHT,
    mode="txt2img",
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=True,
    use_tiny_vae=True,
    acceleration="none",
    cfg_type="none",
    do_add_noise=True,
    warmup=5,
    seed=-1,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=False,
    use_ipadapter=False,
)
wrapper.prepare(prompt=INITIAL_PROMPT, num_inference_steps=50, guidance_scale=1.0)
print(f"Model loaded! Initial prompt: '{INITIAL_PROMPT}'")

# === SHARED MEMORY OUTPUT ===
frame_size = WIDTH * HEIGHT * 3
try:
    out_mem = shared_memory.SharedMemory(name=OUTPUT_MEM_NAME)
    print(f"Connected to existing SharedMemory: {OUTPUT_MEM_NAME}")
except FileNotFoundError:
    out_mem = shared_memory.SharedMemory(name=OUTPUT_MEM_NAME, create=True, size=frame_size)
    print(f"Created SharedMemory: {OUTPUT_MEM_NAME}")

# === OSC HANDLERS ===
osc_client = udp_client.SimpleUDPClient("127.0.0.1", OSC_SEND_PORT)

def on_prompt(addr, *args):
    new_prompt = str(args[0])
    print(f"\n>>> PROMPT: '{new_prompt}'")
    wrapper.stream._param_updater.update_stream_params(
        prompt_list=[(new_prompt, 1.0)]
    )

def on_seed(addr, *args):
    seed_val = int(args[0])
    print(f"\n>>> SEED: {seed_val}")
    wrapper.stream._param_updater.update_stream_params(seed=seed_val)

def on_seed_list(addr, *args):
    import json
    seed_data = json.loads(args[0])
    seed_list = [(int(s[0]), float(s[1])) for s in seed_data]
    print(f"\n>>> SEED_LIST: {seed_list}")
    wrapper.stream._param_updater.update_stream_params(seed_list=seed_list)

def on_delta(addr, *args):
    print(f"\n>>> DELTA: {args[0]}")
    wrapper.stream._param_updater.update_stream_params(delta=float(args[0]))

def on_guidance(addr, *args):
    print(f"\n>>> GUIDANCE: {args[0]}")
    wrapper.stream._param_updater.update_stream_params(guidance_scale=float(args[0]))

def on_t_list(addr, *args):
    t_list = list(args)
    print(f"\n>>> T_LIST: {t_list}")
    wrapper.stream._param_updater.update_stream_params(t_index_list=t_list)

def on_catch_all(addr, *args):
    print(f"[OSC] {addr} = {args}")

dispatcher = Dispatcher()
dispatcher.map("/prompt", on_prompt)
dispatcher.map("/seed", on_seed)
dispatcher.map("/seed_list", on_seed_list)
dispatcher.map("/delta", on_delta)
dispatcher.map("/guidance_scale", on_guidance)
dispatcher.map("/t_list", on_t_list)
dispatcher.set_default_handler(on_catch_all)

osc_server = BlockingOSCUDPServer(("127.0.0.1", OSC_LISTEN_PORT), dispatcher)
osc_thread = threading.Thread(target=osc_server.serve_forever, daemon=True)
osc_thread.start()
print(f"OSC listening on {OSC_LISTEN_PORT}, sending to {OSC_SEND_PORT}")

# === AUTO PROMPT CHANGE TEST ===
import random
prompt_list = [
    "psychedelic neon fractal, vibrant colors",
    "underwater coral reef, bioluminescent creatures", 
    "cyberpunk city, neon rain at night",
    "abstract liquid metal, chrome reflections",
    "deep space nebula, stars and galaxies",
]
prompt_change_interval = 90  # Change prompt every 90 frames (~3 sec)

# Save video frames
video_frames = []
max_video_frames = 300  # 10 seconds at 30fps
video_saved = False

print("\n=== STREAMING (auto-changing prompt every 3s, saving 10s video) ===\n")
frame_count = 0
fps_smooth = 0.0
start_time = time.time()

try:
    while True:
        t0 = time.time()

        # Auto-change prompt
        if frame_count > 0 and frame_count % prompt_change_interval == 0:
            new_prompt = prompt_list[(frame_count // prompt_change_interval) % len(prompt_list)]
            print(f"\n>>> AUTO PROMPT [{frame_count}]: '{new_prompt}'")
            wrapper.stream._param_updater.update_stream_params(
                prompt_list=[(new_prompt, 1.0)]
            )

        # Generate frame
        image = wrapper.txt2img()

        # Convert to numpy
        if isinstance(image, Image.Image):
            frame_np = np.array(image)
        elif isinstance(image, list):
            frame_np = np.array(image[0])
        else:
            frame_np = image

        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        elif frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)

        # Write to SharedMemory
        out_array = np.ndarray(frame_np.shape, dtype=np.uint8, buffer=out_mem.buf)
        np.copyto(out_array, frame_np)

        # Collect video frames
        if not video_saved and len(video_frames) < max_video_frames:
            video_frames.append(Image.fromarray(frame_np))
            if len(video_frames) == max_video_frames:
                print(f"\nSaving {max_video_frames} frames as GIF...")
                video_frames[0].save(
                    "X:/td/StreamDiffusion/stream_test.gif",
                    save_all=True, append_images=video_frames[1:],
                    duration=33, loop=0
                )
                print("Saved stream_test.gif!")
                video_saved = True

        # FPS
        dt = time.time() - t0
        fps_instant = 1.0 / dt if dt > 0 else 0
        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
        frame_count += 1

        # Send heartbeat to TD
        osc_client.send_message('/server_active', 1)
        osc_client.send_message('/framecount', frame_count)
        osc_client.send_message('/stream-info/fps', fps_smooth)
        osc_client.send_message('/stream-info/state', 'local_streaming')
        osc_client.send_message('/stream-info/output-name', OUTPUT_MEM_NAME)

        # Print FPS every second
        if frame_count % 30 == 0:
            uptime = int(time.time() - start_time)
            print(f"\rStreaming | FPS: {fps_smooth:.1f} | Frames: {frame_count} | Uptime: {uptime//60:02d}:{uptime%60:02d}  ", end='', flush=True)

except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    out_mem.close()
    try:
        out_mem.unlink()
    except:
        pass
    osc_server.shutdown()
    print("Done.")
