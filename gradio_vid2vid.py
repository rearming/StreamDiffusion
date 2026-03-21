"""
StreamDiffusion Gradio UI — vid2vid with ControlNet + IPAdapter.
All sliders update in real-time (no reload needed except model change).
Settings auto-persist to JSON between sessions.

Launch:  python gradio_vid2vid.py
"""
import os, sys, time, threading, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import gradio as gr

os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

# Defer heavy imports
StreamDiffusionWrapper = None
def _ensure_sd_imported():
    global StreamDiffusionWrapper
    if StreamDiffusionWrapper is None:
        from streamdiffusion import StreamDiffusionWrapper as _W
        StreamDiffusionWrapper = _W

# ============================================================
# Settings persistence
# ============================================================
SETTINGS_DIR = Path("X:/td/StreamDiffusion/gradio_settings")
SETTINGS_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = SETTINGS_DIR / "last_session.json"
PRESETS_DIR = SETTINGS_DIR / "presets"
PRESETS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = SETTINGS_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


def _persist_file(src_path, subdir=""):
    """Copy a file from Gradio temp dir to persistent storage. Returns persistent path."""
    if not src_path or not os.path.exists(src_path):
        return ""
    src = Path(src_path)
    dest_dir = UPLOADS_DIR / subdir if subdir else UPLOADS_DIR
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / src.name
    # Only copy if not already in our persistent dir
    if src.parent != dest_dir:
        shutil.copy2(str(src), str(dest))
    return str(dest)

# Keys that map 1:1 to UI components
SETTING_KEYS = [
    "model_preset", "custom_model_id", "width", "height", "mode", "acceleration",
    "prompt", "negative_prompt",
    "guidance_scale", "num_inference_steps", "delta", "seed", "cfg_type",
    "cn_preset", "cn_custom_model", "cn_preprocessor", "cn_scale", "cn_enabled",
    "ipa_preset", "ipa_scale",
    "video_path", "style_image_path",
    "playback_speed",
    "auto_load_pipeline",
]

DEFAULTS = {
    "model_preset": "sd-turbo",
    "custom_model_id": "",
    "width": 512,
    "height": 512,
    "mode": "img2img",
    "acceleration": "none",
    "prompt": "masterpiece, high quality, detailed",
    "negative_prompt": "blurry, low quality, distorted",
    "guidance_scale": 1.0,
    "num_inference_steps": 50,
    "delta": 0.5,
    "seed": -1,
    "cfg_type": "none",
    "cn_preset": "None",
    "cn_custom_model": "",
    "cn_preprocessor": "depth",
    "cn_scale": 0.5,
    "cn_enabled": True,
    "ipa_preset": "None",
    "ipa_scale": 0.6,
    "video_path": "",
    "style_image_path": "",
    "playback_speed": 1.0,
    # Settings tab
    "auto_load_pipeline": False,
}

def _load_settings(path=None):
    p = Path(path) if path else SETTINGS_FILE
    if p.exists():
        try:
            with open(p) as f:
                saved = json.load(f)
            merged = {**DEFAULTS, **saved}
            return merged
        except Exception:
            pass
    return dict(DEFAULTS)

def _save_settings(settings, path=None):
    p = Path(path) if path else SETTINGS_FILE
    try:
        with open(p, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Failed to save settings: {e}")

def _collect_settings(*args):
    """Build settings dict from UI component values (order must match SETTING_KEYS)."""
    s = {}
    for i, k in enumerate(SETTING_KEYS):
        if i < len(args):
            s[k] = args[i]
        else:
            s[k] = DEFAULTS.get(k)
    # Type coercion
    for k in ("width", "height", "num_inference_steps", "seed"):
        if k in s and s[k] is not None:
            s[k] = int(s[k])
    for k in ("guidance_scale", "delta", "cn_scale", "ipa_scale", "playback_speed"):
        if k in s and s[k] is not None:
            s[k] = float(s[k])
    for k in ("cn_enabled", "auto_load_pipeline"):
        if k in s:
            s[k] = bool(s[k])
    for k in ("video_path", "style_image_path"):
        if k in s:
            s[k] = s[k] or ""
    return s

# Auto-save on any change
def _autosave(*args):
    keys = SETTING_KEYS
    if len(args) >= len(keys):
        s = {}
        for i, k in enumerate(keys):
            s[k] = args[i]
        _save_settings(s)
    return gr.update()

def save_preset(preset_name, *all_setting_values):
    if not preset_name or not preset_name.strip():
        return "Enter a preset name first", gr.update()
    s = _collect_settings(*all_setting_values)
    path = PRESETS_DIR / f"{preset_name.strip()}.json"
    _save_settings(s, path)
    presets = list_presets()
    return f"Saved: {path.name}", gr.update(choices=presets)

def load_preset(preset_name):
    if not preset_name:
        return [gr.update()] * len(SETTING_KEYS) + ["No preset selected"]
    path = PRESETS_DIR / f"{preset_name}.json"
    if not path.exists():
        return [gr.update()] * len(SETTING_KEYS) + [f"Preset not found: {preset_name}"]
    s = _load_settings(path)
    _save_settings(s)  # also update last_session
    outputs = []
    for k in SETTING_KEYS:
        outputs.append(s.get(k, DEFAULTS.get(k)))
    outputs.append(f"Loaded: {preset_name}")
    return outputs

def delete_preset(preset_name):
    if not preset_name:
        return "No preset selected", gr.update()
    path = PRESETS_DIR / f"{preset_name}.json"
    if path.exists():
        path.unlink()
    presets = list_presets()
    return f"Deleted: {preset_name}", gr.update(choices=presets)

def list_presets():
    return [p.stem for p in sorted(PRESETS_DIR.glob("*.json"))]

# ============================================================
# Presets (model/controlnet/ipadapter)
# ============================================================
MODEL_PRESETS = {
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "t_index_list": [1],
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "use_lcm_lora": True,
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "t_index_list": [1],
        "cfg_type": "none",
        "guidance_scale": 1.0,
        "use_lcm_lora": False,
    },
    "sdxl-base": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "t_index_list": [16, 32],
        "cfg_type": "self",
        "guidance_scale": 1.1,
        "use_lcm_lora": True,
    },
}

CONTROLNET_PRESETS = {
    "None": None,
    "depth-sdxl (xinsir)": {
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "preprocessor": "depth",
    },
    "canny (custom model)": {
        "model_id": "",
        "preprocessor": "canny",
    },
}

IPADAPTER_PRESETS = {
    "None": None,
    "IP-Adapter SD1.5": {
        "type": "regular",
        "ipadapter_model_path": "h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    },
    "IP-Adapter Plus SD1.5": {
        "type": "plus",
        "ipadapter_model_path": "h94/IP-Adapter/models/ip-adapter-plus_sd15.bin",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    },
}

CFG_TYPES = ["none", "self", "full", "initialize"]
ACCELERATION_TYPES = ["none", "xformers", "tensorrt"]
PREPROCESSORS = ["none", "canny", "depth", "lineart", "soft_edge", "openpose", "passthrough"]

# ============================================================
# Pipeline state
# ============================================================
class PipelineState:
    def __init__(self):
        self.wrapper = None
        self.running = False
        self.paused = False
        self.width = 512
        self.height = 512
        self.playback_speed = 1.0
        self.seek_request = None  # set to 0.0-1.0 to seek

state = PipelineState()

def _updater():
    if state.wrapper and hasattr(state.wrapper, 'stream'):
        return state.wrapper.stream._param_updater
    return None

# ============================================================
# Live update functions (no reload)
# ============================================================
def live_prompt(prompt):
    # Go through wrapper to handle text encoder reload/offload for TRT
    if state.wrapper:
        state.wrapper.update_stream_params(prompt_list=[(prompt, 1.0)])
    return gr.update()

def live_negative(neg):
    # Negative prompt also needs text encoder for CFG modes
    if state.wrapper:
        state.wrapper.update_stream_params(negative_prompt=neg)
    return gr.update()

def live_guidance(val):
    u = _updater()
    if u:
        u.update_stream_params(guidance_scale=float(val))
    return gr.update()

def live_delta(val):
    u = _updater()
    if u:
        u.update_stream_params(delta=float(val))
    return gr.update()

def live_seed(val):
    u = _updater()
    if u:
        u.update_stream_params(seed=int(val))
    return gr.update()

def live_steps(val):
    u = _updater()
    if u:
        u.update_stream_params(num_inference_steps=int(val))
    return gr.update()

def live_cn_scale(val):
    if state.wrapper and hasattr(state.wrapper.stream, '_controlnet_module'):
        cn = state.wrapper.stream._controlnet_module
        if cn:
            cn.update_controlnet_scale(0, float(val))
    return gr.update()

def live_cn_enabled(val):
    if state.wrapper and hasattr(state.wrapper.stream, '_controlnet_module'):
        cn = state.wrapper.stream._controlnet_module
        if cn:
            cn.update_controlnet_enabled(0, bool(val))
    return gr.update()

def live_playback_speed(val):
    state.playback_speed = float(val)
    return gr.update()

def live_video_seek(val):
    state.seek_request = float(val)
    return gr.update()

def live_ipa_scale(val):
    if state.wrapper and hasattr(state.wrapper.stream, 'ipadapter'):
        try:
            ipa = state.wrapper.stream.ipadapter
            ipa.set_scale(float(val))
            ipa.scale = float(val)
        except Exception:
            pass
    return gr.update()

def live_style_image(img):
    if state.wrapper is None or img is None:
        return "No pipeline / no image"
    try:
        pil = Image.fromarray(img).resize(
            (state.width, state.height), Image.Resampling.LANCZOS
        )
        state.wrapper.update_style_image(pil, is_stream=False)
        return "Style image updated"
    except Exception as e:
        return f"Error: {e}"

# ============================================================
# Pipeline load
# ============================================================
def load_pipeline(
    model_preset, custom_model_id,
    width, height,
    mode, acceleration,
    num_inference_steps, guidance_scale, delta,
    cfg_type, seed,
    prompt, negative_prompt,
    cn_preset, cn_custom_model, cn_preprocessor, cn_scale,
    ipa_preset, ipa_scale,
    style_image,
):
    if state.wrapper is not None:
        import torch, gc
        # Force full cleanup of old pipeline
        try:
            if hasattr(state.wrapper, 'stream'):
                del state.wrapper.stream
        except Exception:
            pass
        del state.wrapper
        state.wrapper = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    state.width = int(width)
    state.height = int(height)

    if model_preset in MODEL_PRESETS:
        mcfg = MODEL_PRESETS[model_preset]
        model_id = mcfg["model_id"]
        t_index = mcfg["t_index_list"]
        use_lcm = mcfg["use_lcm_lora"]
        default_cfg = mcfg["cfg_type"]
    else:
        model_id = custom_model_id or model_preset
        t_index = [1] if "turbo" in model_id.lower() else [16, 32]
        use_lcm = "turbo" not in model_id.lower()
        default_cfg = "none" if "turbo" in model_id.lower() else "self"

    final_cfg = cfg_type if cfg_type != "auto" else default_cfg

    use_controlnet = cn_preset != "None"
    controlnet_config = None
    if use_controlnet:
        preset = CONTROLNET_PRESETS.get(cn_preset)
        cn_mid = cn_custom_model if cn_custom_model else (preset["model_id"] if preset else "")
        if not cn_mid:
            return "ERROR: ControlNet model ID required"
        cn_pp = {}
        if cn_preprocessor == "depth":
            cn_pp = {"detect_resolution": state.width, "image_resolution": state.width}
        elif cn_preprocessor == "canny":
            cn_pp = {"low_threshold": 100, "high_threshold": 200}
        controlnet_config = [{
            "model_id": cn_mid,
            "conditioning_scale": float(cn_scale),
            "preprocessor": cn_preprocessor,
            "preprocessor_params": cn_pp,
            "enabled": True,
        }]

    use_ipa = ipa_preset != "None"
    ipa_config = None
    if use_ipa:
        preset = IPADAPTER_PRESETS.get(ipa_preset)
        if preset:
            ipa_config = [{**preset, "scale": float(ipa_scale), "enabled": True}]

    try:
        _ensure_sd_imported()

        wrapper = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            t_index_list=t_index,
            width=state.width, height=state.height,
            mode=mode,
            frame_buffer_size=1,
            use_denoising_batch=True,
            use_lcm_lora=use_lcm,
            use_tiny_vae=True,
            acceleration=acceleration,
            cfg_type=final_cfg,
            do_add_noise=True,
            warmup=5,
            seed=int(seed),
            use_safety_checker=False,
            engine_dir="X:/td/StreamDiffusion/engines/td",
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            use_ipadapter=use_ipa,
            ipadapter_config=ipa_config,
        )
        wrapper.prepare(
            prompt=prompt or "masterpiece, high quality",
            negative_prompt=negative_prompt or "",
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
        )

        if hasattr(wrapper.stream, 'delta'):
            wrapper.stream.delta = float(delta)

        state.wrapper = wrapper

        if use_ipa and style_image is not None:
            pil_s = Image.fromarray(style_image).resize(
                (state.width, state.height), Image.Resampling.LANCZOS
            )
            wrapper.update_style_image(pil_s, is_stream=False)

        return f"Loaded: {model_id} ({state.width}x{state.height}) | CN={use_controlnet} IPA={use_ipa}"
    except Exception as e:
        import traceback
        return f"ERROR:\n{traceback.format_exc()}"

# ============================================================
# Output conversion
# ============================================================
def _to_numpy(output):
    if isinstance(output, Image.Image):
        return np.array(output)
    elif isinstance(output, list):
        return np.array(output[0])
    else:
        import torch
        if isinstance(output, torch.Tensor):
            t = output
            if t.dim() == 4:
                t = t[0]
            return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return output

# ============================================================
# Single frame / Streaming
# ============================================================
def process_single_frame(video_path):
    if state.wrapper is None:
        return None, None, "Pipeline not loaded"
    if not video_path:
        return None, None, "No video"

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None, "Cannot read frame"

    fr = cv2.resize(frame, (state.width, state.height))
    fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(fr_rgb)

    if state.wrapper.use_controlnet:
        state.wrapper.update_control_image(0, pil)

    t0 = time.time()
    if state.wrapper.mode == "img2img":
        out = state.wrapper.img2img(pil)
    else:
        out = state.wrapper.txt2img()
    dt = time.time() - t0

    return fr_rgb, _to_numpy(out), f"{dt*1000:.0f}ms ({1/dt:.1f} FPS)"


def stream_video(video_path):
    import torch, gc

    if state.wrapper is None:
        yield None, None, "Pipeline not loaded", gr.update()
        return
    if not video_path:
        yield None, None, "No video", gr.update()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, None, f"Cannot open: {video_path}", gr.update()
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state.running = True
    state.paused = False
    fps_smooth = 0.0
    idx = 0
    frame_accum = 0.0  # fractional frame counter for speed control

    try:
        while state.running:
            if state.paused:
                time.sleep(0.05)
                continue

            # Handle seek requests
            if state.seek_request is not None:
                seek_frac = state.seek_request
                state.seek_request = None
                target_frame = int(seek_frac * max(1, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                idx = target_frame

            # Speed control: skip frames for speed > 1x
            speed = state.playback_speed
            frame_accum += speed
            frames_to_skip = int(frame_accum) - 1  # -1 because we read one below
            frame_accum -= int(frame_accum)

            # Skip frames for fast playback
            for _ in range(frames_to_skip):
                ret = cap.grab()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    break
                idx += 1

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                idx = 0

            t0 = time.time()
            fr = cv2.resize(frame, (state.width, state.height))
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(fr_rgb)

            t_cn = time.time()
            if state.wrapper.use_controlnet:
                state.wrapper.update_control_image(0, pil)
            dt_cn = time.time() - t_cn

            if state.wrapper.use_ipadapter:
                state.wrapper.update_style_image(pil, is_stream=True)

            t_infer = time.time()
            if state.wrapper.mode == "img2img":
                out = state.wrapper.img2img(pil)
            else:
                out = state.wrapper.txt2img()
            dt_infer = time.time() - t_infer

            t_post = time.time()
            out_np = _to_numpy(out)
            dt_post = time.time() - t_post

            dt = time.time() - t0
            fps_i = 1.0 / dt if dt > 0 else 0
            fps_smooth = fps_smooth * 0.9 + fps_i * 0.1
            idx += 1

            # For slow playback (speed < 1x), delay to match target frame rate
            if speed < 1.0:
                target_dt = 1.0 / (video_fps * speed)
                if dt < target_dt:
                    time.sleep(target_dt - dt)

            pos = idx / max(1, total)

            if idx % 10 == 1:
                vram_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"[PERF #{idx}] total={dt*1000:.0f}ms cn={dt_cn*1000:.0f}ms infer={dt_infer*1000:.0f}ms post={dt_post*1000:.0f}ms speed={speed}x VRAM={vram_mb:.0f}MB", flush=True)

            # Periodic cleanup: release PyTorch cached (unused) GPU memory
            if idx % 100 == 0:
                torch.cuda.empty_cache()

            yield (fr_rgb, out_np,
                   f"Frame {idx}/{total} | {fps_smooth:.1f} FPS | {dt*1000:.0f}ms | {speed}x",
                   gr.update(value=min(1.0, pos)))

            # Free intermediates so GC doesn't have to chase them
            del frame, fr, fr_rgb, pil, out, out_np
    finally:
        cap.release()
        state.running = False


def unload_pipeline():
    """Free all GPU/CPU memory from the current pipeline."""
    if state.wrapper is None:
        return "No pipeline loaded"
    import torch, gc
    state.running = False
    try:
        if hasattr(state.wrapper, 'stream'):
            del state.wrapper.stream
    except Exception:
        pass
    del state.wrapper
    state.wrapper = None
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    allocated = torch.cuda.memory_allocated() / 1024**3
    return f"Unloaded. GPU: {allocated:.2f}GB allocated"

def stop_stream():
    state.running = False
    return "Stopped"

def pause_stream():
    state.paused = not state.paused
    return "Paused" if state.paused else "Running"

# ============================================================
# UI
# ============================================================
def _build_generate_tab(s):
    """Build the Generate tab contents. Returns dict of components."""
    c = {}  # components

    with gr.Row():
        # ---- LEFT: Controls ----
        with gr.Column(scale=1, min_width=380):
            with gr.Accordion("Model \u26a0\ufe0f reload required", open=True):
                c["model_preset"] = gr.Dropdown(
                    list(MODEL_PRESETS.keys()), value=s["model_preset"], label="Preset")
                c["custom_model_id"] = gr.Textbox(
                    value=s["custom_model_id"],
                    label="Custom Model ID", placeholder="overrides preset")
                with gr.Row():
                    c["width"] = gr.Number(value=s["width"], label="W", precision=0)
                    c["height"] = gr.Number(value=s["height"], label="H", precision=0)
                with gr.Row():
                    c["mode"] = gr.Radio(["img2img", "txt2img"], value=s["mode"], label="Mode")
                    c["acceleration"] = gr.Dropdown(
                        ACCELERATION_TYPES, value=s["acceleration"], label="Accel")

            with gr.Accordion("Prompt", open=True):
                c["prompt"] = gr.Textbox(value=s["prompt"], label="Prompt", lines=2)
                c["negative_prompt"] = gr.Textbox(
                    value=s["negative_prompt"], label="Negative Prompt")

            with gr.Accordion("Generation (live)", open=True):
                c["guidance_scale"] = gr.Slider(
                    0.0, 20.0, value=s["guidance_scale"], step=0.1, label="Guidance Scale")
                c["num_inference_steps"] = gr.Slider(
                    1, 100, value=s["num_inference_steps"], step=1,
                    label="Inference Steps \u26a0\ufe0f reload")
                c["delta"] = gr.Slider(
                    0.0, 1.0, value=s["delta"], step=0.05,
                    label="Delta (higher = more input preserved)")
                c["seed"] = gr.Slider(
                    0, 1000, value=max(0, s["seed"]), step=1, label="Seed")
                c["cfg_type"] = gr.Dropdown(
                    CFG_TYPES, value=s["cfg_type"],
                    label="CFG Type \u26a0\ufe0f reload")

            with gr.Accordion("ControlNet", open=False):
                c["cn_preset"] = gr.Dropdown(
                    list(CONTROLNET_PRESETS.keys()), value=s["cn_preset"],
                    label="Preset \u26a0\ufe0f reload")
                c["cn_custom_model"] = gr.Textbox(
                    value=s["cn_custom_model"],
                    label="Model ID override \u26a0\ufe0f reload",
                    placeholder="e.g. lllyasviel/...")
                c["cn_preprocessor"] = gr.Dropdown(
                    PREPROCESSORS, value=s["cn_preprocessor"],
                    label="Preprocessor \u26a0\ufe0f reload")
                c["cn_scale"] = gr.Slider(
                    0.0, 2.0, value=s["cn_scale"], step=0.05, label="CN Scale (live)")
                c["cn_enabled"] = gr.Checkbox(value=s["cn_enabled"], label="Enabled (live)")

            with gr.Accordion("IPAdapter", open=False):
                c["ipa_preset"] = gr.Dropdown(
                    list(IPADAPTER_PRESETS.keys()), value=s["ipa_preset"],
                    label="Preset \u26a0\ufe0f reload")
                c["ipa_scale"] = gr.Slider(
                    0.0, 2.0, value=s["ipa_scale"], step=0.05, label="IPA Scale (live)")
                _restored_style = None
                if s["style_image_path"] and os.path.exists(s["style_image_path"]):
                    try:
                        _restored_style = np.array(Image.open(s["style_image_path"]))
                    except Exception:
                        pass
                c["style_image"] = gr.Image(
                    label="Style / Reference Image", type="numpy", value=_restored_style)
                c["style_image_path"] = gr.Textbox(
                    value=s["style_image_path"], visible=False)

            with gr.Row():
                c["load_btn"] = gr.Button("Load Pipeline", variant="primary", size="lg")
                c["unload_btn"] = gr.Button("Unload", variant="stop")
            c["load_status"] = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("Presets", open=False):
                with gr.Row():
                    c["preset_name_input"] = gr.Textbox(
                        label="Preset Name", placeholder="my_setup")
                    c["save_preset_btn"] = gr.Button("Save")
                with gr.Row():
                    c["preset_list"] = gr.Dropdown(
                        choices=list_presets(), label="Load Preset",
                        allow_custom_value=True)
                    c["load_preset_btn"] = gr.Button("Load")
                    c["delete_preset_btn"] = gr.Button("Delete", variant="stop")
                c["preset_status"] = gr.Textbox(label="", interactive=False)

        # ---- RIGHT: Video + Output ----
        with gr.Column(scale=2):
            _restored_video = s["video_path"] if s["video_path"] and os.path.exists(s["video_path"]) else None
            c["video_input"] = gr.Video(label="Input Video", value=_restored_video)
            c["video_path_store"] = gr.Textbox(value=s["video_path"], visible=False)

            with gr.Row():
                c["stream_btn"] = gr.Button("Stream", variant="primary")
                c["single_btn"] = gr.Button("Single Frame")
                c["pause_btn"] = gr.Button("Pause/Resume")
                c["stop_btn"] = gr.Button("Stop", variant="stop")

            with gr.Row():
                c["video_position"] = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.001,
                    label="Position", scale=4)
                c["playback_speed"] = gr.Slider(
                    0.25, 4.0, value=s.get("playback_speed", 1.0),
                    step=0.25, label="Speed", scale=1)

            c["stream_status"] = gr.Textbox(label="", interactive=False)

            with gr.Row():
                c["input_preview"] = gr.Image(label="Input")
                c["output_preview"] = gr.Image(label="Output")

    return c


def build_ui():
    saved = _load_settings()
    s = saved

    with gr.Blocks(title="StreamDiffusion vid2vid") as app:
        gr.Markdown("## StreamDiffusion vid2vid")

        with gr.Tabs():
            with gr.Tab("Generate"):
                c = _build_generate_tab(s)

            with gr.Tab("Settings"):
                gr.Markdown("### App Settings")
                c["auto_load_pipeline"] = gr.Checkbox(
                    value=s.get("auto_load_pipeline", False),
                    label="Auto-load pipeline on start",
                    info="Automatically load the pipeline with saved settings when the app starts",
                )
                gr.Markdown("---")
                gr.Markdown("### Paths")
                gr.Markdown(f"- **Settings:** `{SETTINGS_FILE}`")
                gr.Markdown(f"- **Presets:** `{PRESETS_DIR}`")
                gr.Markdown(f"- **Uploads:** `{UPLOADS_DIR}`")
                settings_status = gr.Textbox(label="", interactive=False)

                def save_auto_load(val):
                    cur = _load_settings()
                    cur["auto_load_pipeline"] = bool(val)
                    _save_settings(cur)
                    return f"Auto-load {'enabled' if val else 'disabled'}"

                c["auto_load_pipeline"].change(
                    save_auto_load,
                    inputs=[c["auto_load_pipeline"]],
                    outputs=[settings_status],
                )

        # ============================================================
        # All saveable components (order must match SETTING_KEYS)
        # ============================================================
        all_settings = [
            c["model_preset"], c["custom_model_id"], c["width"], c["height"],
            c["mode"], c["acceleration"],
            c["prompt"], c["negative_prompt"],
            c["guidance_scale"], c["num_inference_steps"], c["delta"],
            c["seed"], c["cfg_type"],
            c["cn_preset"], c["cn_custom_model"], c["cn_preprocessor"],
            c["cn_scale"], c["cn_enabled"],
            c["ipa_preset"], c["ipa_scale"],
            c["video_path_store"], c["style_image_path"],
            c["playback_speed"],
            c["auto_load_pipeline"],
        ]

        # Auto-save on any change
        for comp in all_settings:
            comp.change(_autosave, inputs=all_settings, outputs=[])

        # Track video: copy to persistent storage
        def on_video_change(vid):
            return _persist_file(vid, "videos") if vid else ""
        c["video_input"].change(
            on_video_change, inputs=[c["video_input"]], outputs=[c["video_path_store"]])

        # Track style image
        def on_style_change(img):
            if img is not None:
                path = UPLOADS_DIR / "style_images" / "last_style.png"
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(img).save(str(path))
                return str(path)
            return ""
        c["style_image"].change(
            on_style_change, inputs=[c["style_image"]], outputs=[c["style_image_path"]])

        # Live controls
        c["prompt"].change(live_prompt, inputs=[c["prompt"]], outputs=[])
        c["negative_prompt"].change(live_negative, inputs=[c["negative_prompt"]], outputs=[])
        c["guidance_scale"].release(live_guidance, inputs=[c["guidance_scale"]], outputs=[])
        c["num_inference_steps"].release(live_steps, inputs=[c["num_inference_steps"]], outputs=[])
        c["delta"].release(live_delta, inputs=[c["delta"]], outputs=[])
        c["seed"].release(live_seed, inputs=[c["seed"]], outputs=[])
        c["playback_speed"].release(live_playback_speed, inputs=[c["playback_speed"]], outputs=[])
        c["video_position"].release(live_video_seek, inputs=[c["video_position"]], outputs=[])
        c["cn_scale"].release(live_cn_scale, inputs=[c["cn_scale"]], outputs=[])
        c["cn_enabled"].change(live_cn_enabled, inputs=[c["cn_enabled"]], outputs=[])
        c["ipa_scale"].release(live_ipa_scale, inputs=[c["ipa_scale"]], outputs=[])
        c["style_image"].change(
            live_style_image, inputs=[c["style_image"]], outputs=[c["load_status"]])

        # Pipeline load inputs (reused)
        _load_inputs = [
            c["model_preset"], c["custom_model_id"],
            c["width"], c["height"],
            c["mode"], c["acceleration"],
            c["num_inference_steps"], c["guidance_scale"], c["delta"],
            c["cfg_type"], c["seed"],
            c["prompt"], c["negative_prompt"],
            c["cn_preset"], c["cn_custom_model"], c["cn_preprocessor"], c["cn_scale"],
            c["ipa_preset"], c["ipa_scale"],
            c["style_image"],
        ]

        c["load_btn"].click(fn=load_pipeline, inputs=_load_inputs, outputs=[c["load_status"]])
        c["unload_btn"].click(fn=unload_pipeline, outputs=[c["load_status"]])

        # Streaming
        c["single_btn"].click(
            fn=process_single_frame, inputs=[c["video_input"]],
            outputs=[c["input_preview"], c["output_preview"], c["stream_status"]])
        c["stream_btn"].click(
            fn=stream_video, inputs=[c["video_input"]],
            outputs=[c["input_preview"], c["output_preview"], c["stream_status"],
                     c["video_position"]])
        c["stop_btn"].click(fn=stop_stream, outputs=[c["stream_status"]])
        c["pause_btn"].click(fn=pause_stream, outputs=[c["stream_status"]])

        # Presets
        c["save_preset_btn"].click(
            fn=save_preset,
            inputs=[c["preset_name_input"]] + all_settings,
            outputs=[c["preset_status"], c["preset_list"]])
        c["load_preset_btn"].click(
            fn=load_preset, inputs=[c["preset_list"]],
            outputs=all_settings + [c["preset_status"]])
        c["delete_preset_btn"].click(
            fn=delete_preset, inputs=[c["preset_list"]],
            outputs=[c["preset_status"], c["preset_list"]])

        # Auto-load pipeline on start
        if s.get("auto_load_pipeline", False):
            app.load(fn=load_pipeline, inputs=_load_inputs, outputs=[c["load_status"]])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
