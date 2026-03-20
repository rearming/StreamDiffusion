"""
StreamDiffusion preview: video-to-video with ControlNet + optional IPAdapter.
Reads frames from a video file, transforms them through SD pipeline,
shows side-by-side preview in an OpenCV window.

Usage:
  # Basic img2img with sd-turbo (default)
  python server_preview_vid2vid.py --video path/to/video.mp4

  # Use sdxl-turbo
  python server_preview_vid2vid.py --video path/to/video.mp4 --model sdxl-turbo

  # With ControlNet (depth, SDXL)
  python server_preview_vid2vid.py --video path/to/video.mp4 --model sdxl-turbo --controlnet depth

  # With ControlNet canny (any model, no extra downloads)
  python server_preview_vid2vid.py --video path/to/video.mp4 --controlnet canny --controlnet-model <model-id>

Controls:
  Q / window X   - quit
  SPACE          - pause/resume video
  C              - toggle ControlNet on/off
  +/-            - adjust ControlNet scale
  [ / ]          - adjust preprocessor params
  P              - cycle through prompts
  S              - toggle IPAdapter style source (video/static)
"""
import os, sys, time, threading, argparse
import numpy as np
from PIL import Image
import cv2

os.environ['HF_HOME'] = 'X:/hf_cache'
# Uncomment these after first run to skip network checks:
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# === MODEL PRESETS ===
MODEL_PRESETS = {
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "t_index_list": [1],
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "use_lcm_lora": True,
        "family": "sd21",
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "t_index_list": [1],
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "use_lcm_lora": False,  # sdxl-turbo doesn't need LCM LoRA
        "family": "sdxl",
    },
    "sdxl-base": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "t_index_list": [16, 32],
        "cfg_type": "self",
        "guidance_scale": 1.1,
        "use_lcm_lora": True,
        "family": "sdxl",
    },
}

# === CONTROLNET PRESETS ===
CONTROLNET_PRESETS = {
    "depth-sdxl": {
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "preprocessor": "depth",  # needs depth estimator
        "family": "sdxl",
    },
    "canny": {
        "model_id": None,  # must be specified via --controlnet-model
        "preprocessor": "canny",
        "family": "any",
    },
    "none": None,
}

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description="StreamDiffusion vid2vid preview")
parser.add_argument("--video", type=str, required=True, help="Input video file path")
parser.add_argument("--style", type=str, default=None, help="Static style image for IPAdapter")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
# Model
parser.add_argument("--model", type=str, default="sd-turbo",
                    help=f"Model preset ({', '.join(MODEL_PRESETS.keys())}) or HuggingFace model ID")
parser.add_argument("--acceleration", choices=["none", "xformers", "tensorrt"], default="none")
parser.add_argument("--mode", choices=["img2img", "txt2img"], default="img2img")
parser.add_argument("--delta", type=float, default=0.5, help="Residual noise delta")
parser.add_argument("--prompt", type=str, default="masterpiece, high quality, detailed")
# ControlNet
parser.add_argument("--controlnet", type=str, default="none",
                    help=f"ControlNet preset ({', '.join(CONTROLNET_PRESETS.keys())}) or 'none'")
parser.add_argument("--controlnet-model", type=str, default=None, help="Override ControlNet model ID")
parser.add_argument("--controlnet-scale", type=float, default=0.5, help="ControlNet conditioning scale")
parser.add_argument("--canny-low", type=int, default=100, help="Canny low threshold")
parser.add_argument("--canny-high", type=int, default=200, help="Canny high threshold")
# IPAdapter
parser.add_argument("--no-ipadapter", action="store_true", help="Disable IPAdapter")
parser.add_argument("--ipadapter-scale", type=float, default=0.6, help="IPAdapter conditioning scale")
# Other
parser.add_argument("--osc-port", type=int, default=8577)
args = parser.parse_args()

WIDTH, HEIGHT = args.width, args.height

# Resolve model preset
if args.model in MODEL_PRESETS:
    model_cfg = MODEL_PRESETS[args.model]
    print(f"Using model preset: {args.model}")
else:
    # Custom model ID - guess settings
    model_cfg = {
        "model_id": args.model,
        "t_index_list": [1] if "turbo" in args.model.lower() else [16, 32],
        "cfg_type": "none" if "turbo" in args.model.lower() else "self",
        "guidance_scale": 1.0,
        "use_lcm_lora": "turbo" not in args.model.lower(),
        "family": "sdxl" if "xl" in args.model.lower() else "sd15",
    }
    print(f"Using custom model: {args.model}")

# Resolve ControlNet
USE_CONTROLNET = args.controlnet != "none"
controlnet_config = None
cn_preprocessor = "none"

if USE_CONTROLNET:
    if args.controlnet in CONTROLNET_PRESETS and CONTROLNET_PRESETS[args.controlnet]:
        cn_preset = CONTROLNET_PRESETS[args.controlnet]
        cn_model_id = args.controlnet_model or cn_preset["model_id"]
        cn_preprocessor = cn_preset["preprocessor"]
    else:
        cn_model_id = args.controlnet_model or args.controlnet
        cn_preprocessor = "canny"  # default

    if not cn_model_id:
        print(f"ERROR: ControlNet preset '{args.controlnet}' needs --controlnet-model")
        sys.exit(1)

    cn_preproc_params = {}
    if cn_preprocessor == "depth":
        cn_preproc_params = {"detect_resolution": WIDTH, "image_resolution": WIDTH}
    elif cn_preprocessor == "canny":
        cn_preproc_params = {"low_threshold": args.canny_low, "high_threshold": args.canny_high}

    controlnet_config = [{
        "model_id": cn_model_id,
        "conditioning_scale": args.controlnet_scale,
        "preprocessor": cn_preprocessor,
        "preprocessor_params": cn_preproc_params,
        "enabled": True,
    }]
    print(f"ControlNet: {cn_model_id} (preprocessor: {cn_preprocessor}, scale: {args.controlnet_scale})")

# IPAdapter
USE_IPADAPTER = not args.no_ipadapter
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

# === LOAD STYLE IMAGE ===
style_image = None
if args.style:
    style_image = Image.open(args.style).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    print(f"Loaded style image: {args.style}")

# === LOAD MODEL ===
print(f"Loading {model_cfg['model_id']}...")
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path=model_cfg["model_id"],
    t_index_list=model_cfg["t_index_list"],
    width=WIDTH, height=HEIGHT,
    mode=args.mode,
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=model_cfg["use_lcm_lora"],
    use_tiny_vae=True,
    acceleration=args.acceleration,
    cfg_type=model_cfg["cfg_type"],
    do_add_noise=True,
    warmup=5,
    seed=-1,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=USE_CONTROLNET,
    controlnet_config=controlnet_config,
    use_ipadapter=USE_IPADAPTER,
    ipadapter_config=ipadapter_config,
)
wrapper.prepare(
    prompt=args.prompt,
    num_inference_steps=50,
    guidance_scale=model_cfg["guidance_scale"],
)
print("Model loaded!")

# === INITIAL STYLE IMAGE ===
if USE_IPADAPTER:
    if style_image:
        wrapper.update_style_image(style_image, is_stream=False)
        print("IPAdapter: using static style image")
    else:
        first_frame = read_video_frame()
        if first_frame:
            wrapper.update_style_image(first_frame, is_stream=False)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("IPAdapter: using video frames as style source")

# === HELPERS ===
controlnet_enabled = USE_CONTROLNET
current_cn_scale = args.controlnet_scale

def _set_ipa_scale(scale):
    """Update IPAdapter scale at runtime."""
    try:
        ipa = wrapper.stream.ipadapter
        ipa.set_scale(float(scale))
        ipa.scale = float(scale)
    except Exception:
        if hasattr(wrapper.stream, '_ipadapter_module'):
            wrapper.stream._ipadapter_module.config.scale = float(scale)

def _window_closed(name):
    """Check if OpenCV window was closed by user (X button)."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True

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
        _set_ipa_scale(float(args_osc[0]))

def on_controlnet_scale(addr, *args_osc):
    global current_cn_scale
    if USE_CONTROLNET:
        current_cn_scale = float(args_osc[0])

dispatcher = Dispatcher()
dispatcher.map("/prompt", on_prompt)
dispatcher.map("/seed", on_seed)
dispatcher.map("/ipadapter/scale", on_ipadapter_scale)
dispatcher.map("/controlnet/scale", on_controlnet_scale)
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
controls = "Q=quit SPACE=pause P=prompt"
if USE_CONTROLNET:
    controls += " C=toggle-CN +/-=CN-scale"
    if cn_preprocessor == "canny":
        controls += " [/]=canny"
if USE_IPADAPTER:
    controls += " S=style-src"
print(f"\n=== VID2VID PREVIEW ({controls}) ===\n")

window_name = "StreamDiffusion vid2vid"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
panel_gap = 4
display_w = WIDTH * 2 + panel_gap
cv2.resizeWindow(window_name, display_w, HEIGHT)

frame_count = 0
fps_smooth = 0.0
paused = False
use_video_as_style = (style_image is None)
current_ipa_scale = args.ipadapter_scale
video_frame = None

try:
    while True:
        if _window_closed(window_name):
            break

        t0 = time.time()

        # Read video frame
        if not paused:
            video_frame = read_video_frame()
            if video_frame is None:
                print("Video ended / error reading frame")
                break

        # Feed video frame to ControlNet (preprocessor runs internally)
        if USE_CONTROLNET and controlnet_enabled and video_frame is not None:
            wrapper.update_control_image(0, video_frame)

        # Update IPAdapter style
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
        input_bgr = cv2.cvtColor(np.array(video_frame), cv2.COLOR_RGB2BGR)

        # Build combined display (input | output)
        gap = np.zeros((HEIGHT, panel_gap, 3), dtype=np.uint8)
        combined = np.hstack([input_bgr, gap, out_bgr])

        # FPS & info
        dt = time.time() - t0
        fps_instant = 1.0 / dt if dt > 0 else 0
        fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
        frame_count += 1

        vid_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        info_lines = [
            f"FPS: {fps_smooth:.1f}  model: {args.model}",
            f"Frame: {vid_pos}/{video_frame_count}",
        ]
        if USE_CONTROLNET:
            cn_str = f"CN: {current_cn_scale:.2f}" if controlnet_enabled else "CN: OFF"
            if cn_preprocessor == "canny":
                cn_str += f" canny({canny_low},{canny_high})"
            info_lines.append(cn_str)
        if USE_IPADAPTER:
            info_lines.append(f"IPA: {current_ipa_scale:.2f} ({'video' if use_video_as_style else 'static'})")
        if paused:
            info_lines.append("PAUSED")

        for i, line in enumerate(info_lines):
            cv2.putText(combined, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Labels
        cv2.putText(combined, "Input", (WIDTH // 2 - 25, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined, "Output", (WIDTH + panel_gap + WIDTH // 2 - 30, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Prompt
        cv2.putText(combined, current_prompt[:70], (10, HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

        cv2.imshow(window_name, combined)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('c'):
            if USE_CONTROLNET:
                controlnet_enabled = not controlnet_enabled
                print(f"ControlNet: {'ON' if controlnet_enabled else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            if USE_CONTROLNET:
                current_cn_scale = min(current_cn_scale + 0.05, 1.5)
                print(f"ControlNet scale: {current_cn_scale:.2f}")
            elif USE_IPADAPTER:
                current_ipa_scale = min(current_ipa_scale + 0.05, 1.5)
                _set_ipa_scale(current_ipa_scale)
                print(f"IPAdapter scale: {current_ipa_scale:.2f}")
        elif key == ord('-'):
            if USE_CONTROLNET:
                current_cn_scale = max(current_cn_scale - 0.05, 0.0)
                print(f"ControlNet scale: {current_cn_scale:.2f}")
            elif USE_IPADAPTER:
                current_ipa_scale = max(current_ipa_scale - 0.05, 0.0)
                _set_ipa_scale(current_ipa_scale)
                print(f"IPAdapter scale: {current_ipa_scale:.2f}")
        elif key == ord('['):
            if cn_preprocessor == "canny":
                canny_low = max(canny_low - 10, 10)
                canny_high = max(canny_high - 10, canny_low + 10)
                print(f"Canny: {canny_low}/{canny_high}")
        elif key == ord(']'):
            if cn_preprocessor == "canny":
                canny_low = min(canny_low + 10, 250)
                canny_high = min(canny_high + 10, 255)
                print(f"Canny: {canny_low}/{canny_high}")
        elif key == ord('s'):
            if USE_IPADAPTER:
                use_video_as_style = not use_video_as_style
                if not use_video_as_style and style_image:
                    wrapper.update_style_image(style_image, is_stream=False)
                print(f"IPAdapter style: {'video frames' if use_video_as_style else 'static image'}")
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
