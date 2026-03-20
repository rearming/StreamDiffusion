"""
StreamDiffusion preview: video-to-video with IP Adapter.
Reads frames from a video file, transforms them with SD + IPAdapter,
shows side-by-side preview in an OpenCV window.

Usage:
  python server_preview_vid2vid.py --video path/to/video.mp4
  python server_preview_vid2vid.py --video path/to/video.mp4 --style path/to/style.jpg
  python server_preview_vid2vid.py --video path/to/video.mp4 --ipadapter-scale 0.8

Controls:
  Q          - quit
  SPACE      - pause/resume video
  +/-        - adjust IPAdapter scale
  S          - toggle: video frames as IPAdapter style source
  P          - cycle through prompts
"""
import os, sys, time, threading, argparse
import numpy as np
from PIL import Image
import cv2

os.environ['HF_HOME'] = 'X:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="StreamDiffusion vid2vid preview with IPAdapter")
parser.add_argument("--video", type=str, required=True, help="Input video file path")
parser.add_argument("--style", type=str, default=None, help="Static style image for IPAdapter (if not set, video frames are used)")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--ipadapter-scale", type=float, default=0.6, help="IPAdapter conditioning scale")
parser.add_argument("--delta", type=float, default=0.5, help="Residual noise delta (temporal consistency)")
parser.add_argument("--prompt", type=str, default="masterpiece, high quality, detailed", help="Initial prompt")
parser.add_argument("--osc-port", type=int, default=8577)
parser.add_argument("--mode", choices=["img2img", "txt2img"], default="img2img", help="Generation mode")
parser.add_argument("--model", type=str, default="stabilityai/sd-turbo", help="Model ID or path")
parser.add_argument("--acceleration", choices=["none", "xformers", "tensorrt"], default="none")
parser.add_argument("--no-ipadapter", action="store_true", help="Disable IPAdapter (plain img2img)")
args = parser.parse_args()

WIDTH, HEIGHT = args.width, args.height
USE_IPADAPTER = not args.no_ipadapter

# === LOAD VIDEO ===
print(f"Opening video: {args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {args.video}")
    sys.exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {video_frame_count} frames @ {video_fps:.1f} FPS")

def read_video_frame():
    """Read next frame from video, loop if at end. Returns PIL Image."""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return None
    frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# === LOAD STYLE IMAGE (if provided) ===
style_image = None
if args.style:
    style_image = Image.open(args.style).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    print(f"Loaded style image: {args.style}")

# === BUILD IPADAPTER CONFIG ===
ipadapter_config = None
if USE_IPADAPTER:
    ipadapter_config = [{
        "type": "regular",
        "ipadapter_model_path": "h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
        "scale": args.ipadapter_scale,
        "enabled": True,
    }]
    print(f"IPAdapter enabled, scale={args.ipadapter_scale}")

# === LOAD MODEL ===
print(f"Loading {args.model}...")
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path=args.model,
    t_index_list=[1] if "turbo" in args.model.lower() else [16, 32],
    width=WIDTH, height=HEIGHT,
    mode=args.mode,
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=True,
    use_tiny_vae=True,
    acceleration=args.acceleration,
    cfg_type="none" if "turbo" in args.model.lower() else "self",
    do_add_noise=True,
    warmup=5,
    seed=-1,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=False,
    use_ipadapter=USE_IPADAPTER,
    ipadapter_config=ipadapter_config,
)
wrapper.prepare(prompt=args.prompt, num_inference_steps=50, guidance_scale=1.0)
print("Model loaded!")

# === INITIAL STYLE IMAGE ===
if USE_IPADAPTER:
    # Feed initial style image (either static or first video frame)
    if style_image:
        wrapper.update_style_image(style_image, is_stream=False)
        print("IPAdapter: using static style image")
    else:
        first_frame = read_video_frame()
        if first_frame:
            wrapper.update_style_image(first_frame, is_stream=False)
            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("IPAdapter: using video frames as style source")

# === HELPERS ===
def _set_ipa_scale(scale):
    """Update IPAdapter scale at runtime."""
    try:
        ipa = wrapper.stream.ipadapter
        ipa.set_scale(float(scale))
        ipa.scale = float(scale)
    except Exception:
        # Fallback: set on config
        if hasattr(wrapper.stream, '_ipadapter_module'):
            wrapper.stream._ipadapter_module.config.scale = float(scale)

# === OSC HANDLERS ===
current_prompt = args.prompt

def on_prompt(addr, *args_osc):
    global current_prompt
    current_prompt = str(args_osc[0])
    print(f"\n>>> OSC PROMPT: '{current_prompt}'")
    wrapper.stream._param_updater.update_stream_params(
        prompt_list=[(current_prompt, 1.0)]
    )

def on_seed(addr, *args_osc):
    print(f"\n>>> OSC SEED: {args_osc[0]}")
    wrapper.stream._param_updater.update_stream_params(seed=int(args_osc[0]))

def on_ipadapter_scale(addr, *args_osc):
    if USE_IPADAPTER:
        scale = float(args_osc[0])
        print(f"\n>>> OSC IPADAPTER SCALE: {scale}")
        _set_ipa_scale(scale)

dispatcher = Dispatcher()
dispatcher.map("/prompt", on_prompt)
dispatcher.map("/seed", on_seed)
dispatcher.map("/ipadapter/scale", on_ipadapter_scale)
dispatcher.set_default_handler(lambda *a: None)

osc_server = BlockingOSCUDPServer(("127.0.0.1", args.osc_port), dispatcher)
osc_thread = threading.Thread(target=osc_server.serve_forever, daemon=True)
osc_thread.start()
print(f"OSC on port {args.osc_port}")

# === PROMPT CYCLING ===
prompts = [
    args.prompt,
    "psychedelic neon fractal, vibrant colors",
    "underwater coral reef, bioluminescent creatures",
    "cyberpunk city, neon rain at night",
    "abstract liquid metal, chrome reflections",
    "studio ghibli style, detailed anime artwork",
]
prompt_idx = 0

# === MAIN LOOP ===
print(f"\n=== VID2VID PREVIEW (Q=quit, SPACE=pause, +/-=IPAdapter scale, S=toggle style src, P=cycle prompt) ===\n")

window_name = "StreamDiffusion vid2vid"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
display_w = WIDTH * 2 + 20  # side-by-side with gap
cv2.resizeWindow(window_name, display_w, HEIGHT)

frame_count = 0
fps_smooth = 0.0
paused = False
use_video_as_style = (style_image is None)  # toggle for IPAdapter style source
current_ipa_scale = args.ipadapter_scale

try:
    while True:
        t0 = time.time()

        # Read video frame (skip if paused, reuse last)
        if not paused:
            video_frame = read_video_frame()
            if video_frame is None:
                print("Video ended / error reading frame")
                break

        # Update IPAdapter style from video frame (if enabled and using video as source)
        if USE_IPADAPTER and use_video_as_style and not paused:
            wrapper.update_style_image(video_frame, is_stream=True)

        # Generate
        if args.mode == "img2img":
            output = wrapper.img2img(video_frame)
        else:
            output = wrapper.txt2img()

        # Convert output to numpy
        if isinstance(output, Image.Image):
            out_np = np.array(output)
        elif isinstance(output, list):
            out_np = np.array(output[0])
        else:
            # torch tensor
            import torch
            if hasattr(output, 'cpu'):
                t = output
                if t.dim() == 4:
                    t = t[0]
                out_np = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                out_np = output

        if out_np.max() <= 1.0 and out_np.dtype != np.uint8:
            out_np = (out_np * 255).astype(np.uint8)

        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (WIDTH, HEIGHT))

        # Input frame for display
        input_np = np.array(video_frame)
        input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)

        # Side-by-side: input | gap | output
        gap = np.zeros((HEIGHT, 20, 3), dtype=np.uint8)
        combined = np.hstack([input_bgr, gap, out_bgr])

        # Overlay info
        dt = time.time() - t0
        fps_instant = 1.0 / dt if dt > 0 else 0
        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
        frame_count += 1

        vid_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        info_lines = [
            f"FPS: {fps_smooth:.1f}",
            f"Frame: {vid_pos}/{video_frame_count}",
        ]
        if USE_IPADAPTER:
            info_lines.append(f"IPA: {current_ipa_scale:.2f} ({'video' if use_video_as_style else 'static'})")
        if paused:
            info_lines.append("PAUSED")

        for i, line in enumerate(info_lines):
            cv2.putText(combined, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Labels
        cv2.putText(combined, "Input", (WIDTH // 2 - 25, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined, "Output", (WIDTH + 20 + WIDTH // 2 - 30, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Prompt at bottom
        cv2.putText(combined, current_prompt[:60], (10, HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

        cv2.imshow(window_name, combined)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('+') or key == ord('='):
            if USE_IPADAPTER:
                current_ipa_scale = min(current_ipa_scale + 0.05, 1.5)
                _set_ipa_scale(current_ipa_scale)
                print(f"IPAdapter scale: {current_ipa_scale:.2f}")
        elif key == ord('-'):
            if USE_IPADAPTER:
                current_ipa_scale = max(current_ipa_scale - 0.05, 0.0)
                _set_ipa_scale(current_ipa_scale)
                print(f"IPAdapter scale: {current_ipa_scale:.2f}")
        elif key == ord('s'):
            if USE_IPADAPTER:
                use_video_as_style = not use_video_as_style
                if not use_video_as_style and style_image:
                    wrapper.update_style_image(style_image, is_stream=False)
                print(f"IPAdapter style source: {'video frames' if use_video_as_style else 'static image'}")
        elif key == ord('p'):
            prompt_idx = (prompt_idx + 1) % len(prompts)
            current_prompt = prompts[prompt_idx]
            print(f"Prompt: '{current_prompt}'")
            wrapper.stream._param_updater.update_stream_params(
                prompt_list=[(current_prompt, 1.0)]
            )

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    osc_server.shutdown()
    print("\nDone.")
