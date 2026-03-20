"""
StreamDiffusion server with live preview window.
No TD needed - just a window on screen showing the stream.
"""
import os, sys, time, threading
import numpy as np
from PIL import Image
import cv2

os.environ['HF_HOME'] = 'X:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# === CONFIG ===
WIDTH, HEIGHT = 512, 512
OSC_LISTEN_PORT = 8577
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
print(f"Model loaded!")

# === OSC HANDLERS ===
current_prompt = INITIAL_PROMPT

def on_prompt(addr, *args):
    global current_prompt
    current_prompt = str(args[0])
    print(f"\n>>> PROMPT: '{current_prompt}'")
    wrapper.stream._param_updater.update_stream_params(
        prompt_list=[(current_prompt, 1.0)]
    )

def on_seed(addr, *args):
    print(f"\n>>> SEED: {args[0]}")
    wrapper.stream._param_updater.update_stream_params(seed=int(args[0]))

def on_catch_all(addr, *args):
    pass  # silent

dispatcher = Dispatcher()
dispatcher.map("/prompt", on_prompt)
dispatcher.map("/seed", on_seed)
dispatcher.set_default_handler(on_catch_all)

osc_server = BlockingOSCUDPServer(("127.0.0.1", OSC_LISTEN_PORT), dispatcher)
osc_thread = threading.Thread(target=osc_server.serve_forever, daemon=True)
osc_thread.start()
print(f"OSC on port {OSC_LISTEN_PORT}")

# === AUTO PROMPT ROTATION ===
prompts = [
    "psychedelic neon fractal, vibrant colors",
    "underwater coral reef, bioluminescent creatures",
    "cyberpunk city, neon rain at night",
    "abstract liquid metal, chrome reflections",
    "deep space nebula, stars and galaxies",
]
prompt_change_every = 90  # frames

# === MAIN LOOP WITH CV2 WINDOW ===
print("\n=== LIVE PREVIEW (press Q to quit, SPACE to pause auto-prompt) ===\n")

cv2.namedWindow("StreamDiffusion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("StreamDiffusion", 768, 768)

frame_count = 0
fps_smooth = 0.0
auto_prompt = True

try:
    while True:
        t0 = time.time()

        # Auto-change prompt
        if auto_prompt and frame_count > 0 and frame_count % prompt_change_every == 0:
            idx = (frame_count // prompt_change_every) % len(prompts)
            current_prompt = prompts[idx]
            print(f"\n>>> AUTO [{frame_count}]: '{current_prompt}'")
            wrapper.stream._param_updater.update_stream_params(
                prompt_list=[(current_prompt, 1.0)]
            )

        # Generate
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

        # RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Add overlay text
        dt = time.time() - t0
        fps_instant = 1.0 / dt if dt > 0 else 0
        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
        frame_count += 1

        cv2.putText(frame_bgr, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_bgr, current_prompt[:50], (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show
        cv2.imshow("StreamDiffusion", frame_bgr)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            auto_prompt = not auto_prompt
            print(f"\nAuto-prompt: {'ON' if auto_prompt else 'OFF'}")

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    osc_server.shutdown()
    print("\nDone.")
