"""
StreamDiffusion Gradio UI — vid2vid with ControlNet + IPAdapter.

Launch:
  python gradio_vid2vid.py
"""
import os, sys, time, threading
import numpy as np
from PIL import Image
import cv2
import gradio as gr

os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

# Defer heavy imports — only load StreamDiffusion when pipeline is actually created
StreamDiffusionWrapper = None
def _ensure_sd_imported():
    global StreamDiffusionWrapper
    if StreamDiffusionWrapper is None:
        from streamdiffusion import StreamDiffusionWrapper as _W
        StreamDiffusionWrapper = _W

# ============================================================
# Presets
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
        "model_id": "",  # user fills in
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
        self.cap = None
        self.running = False
        self.paused = False
        self.video_path = None
        self.width = 512
        self.height = 512
        self.fps_smooth = 0.0
        self.frame_count = 0
        self.last_output = None
        self.last_input = None
        self.status = "Not loaded"

state = PipelineState()

# ============================================================
# Core functions
# ============================================================
def load_pipeline(
    model_preset, custom_model_id,
    width, height,
    mode, acceleration,
    num_inference_steps, guidance_scale,
    cfg_type,
    prompt, negative_prompt,
    # ControlNet
    cn_preset, cn_custom_model, cn_preprocessor, cn_scale,
    # IPAdapter
    ipa_preset, ipa_scale,
    # Style image
    style_image,
):
    """Load/reload the StreamDiffusion pipeline with given params."""
    # Clean up old pipeline
    if state.wrapper is not None:
        del state.wrapper
        state.wrapper = None
        import torch
        torch.cuda.empty_cache()

    state.width = int(width)
    state.height = int(height)

    # Resolve model
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

    final_cfg_type = cfg_type if cfg_type != "auto" else default_cfg

    # ControlNet config
    use_controlnet = cn_preset != "None"
    controlnet_config = None
    if use_controlnet:
        preset = CONTROLNET_PRESETS.get(cn_preset)
        cn_model_id = cn_custom_model if cn_custom_model else (preset["model_id"] if preset else "")
        if not cn_model_id:
            return "ERROR: ControlNet model ID required"

        cn_preproc_params = {}
        if cn_preprocessor == "depth":
            cn_preproc_params = {"detect_resolution": state.width, "image_resolution": state.width}
        elif cn_preprocessor == "canny":
            cn_preproc_params = {"low_threshold": 100, "high_threshold": 200}

        controlnet_config = [{
            "model_id": cn_model_id,
            "conditioning_scale": float(cn_scale),
            "preprocessor": cn_preprocessor,
            "preprocessor_params": cn_preproc_params,
            "enabled": True,
        }]

    # IPAdapter config
    use_ipadapter = ipa_preset != "None"
    ipadapter_config = None
    if use_ipadapter:
        preset = IPADAPTER_PRESETS.get(ipa_preset)
        if preset:
            ipadapter_config = [{
                **preset,
                "scale": float(ipa_scale),
                "enabled": True,
            }]

    # Build wrapper
    try:
        _ensure_sd_imported()
        state.status = "Loading model..."

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
            cfg_type=final_cfg_type,
            do_add_noise=True,
            warmup=5,
            seed=-1,
            use_safety_checker=False,
            engine_dir="X:/td/StreamDiffusion/engines/td",
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            use_ipadapter=use_ipadapter,
            ipadapter_config=ipadapter_config,
        )
        wrapper.prepare(
            prompt=prompt or "masterpiece, high quality",
            negative_prompt=negative_prompt or "",
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
        )

        state.wrapper = wrapper

        # Set initial style image for IPAdapter
        if use_ipadapter and style_image is not None:
            pil_style = Image.fromarray(style_image).resize(
                (state.width, state.height), Image.Resampling.LANCZOS
            )
            wrapper.update_style_image(pil_style, is_stream=False)

        state.status = f"Loaded: {model_id}"
        return f"Pipeline loaded: {model_id} ({state.width}x{state.height})"

    except Exception as e:
        state.status = f"Error: {e}"
        import traceback
        return f"ERROR loading pipeline:\n{traceback.format_exc()}"


def update_prompt(prompt, negative_prompt):
    """Update prompt without reloading the pipeline."""
    if state.wrapper is None:
        return "Pipeline not loaded"
    try:
        state.wrapper.stream._param_updater.update_stream_params(
            prompt_list=[(prompt, 1.0)]
        )
        return f"Prompt updated"
    except Exception as e:
        return f"Error: {e}"


def update_style_image(style_image):
    """Update IPAdapter style image."""
    if state.wrapper is None:
        return "Pipeline not loaded"
    if style_image is None:
        return "No image provided"
    try:
        pil = Image.fromarray(style_image).resize(
            (state.width, state.height), Image.Resampling.LANCZOS
        )
        state.wrapper.update_style_image(pil, is_stream=False)
        return "Style image updated"
    except Exception as e:
        return f"Error: {e}"


def process_single_frame(video_path):
    """Process a single frame from video (for testing / non-streaming use)."""
    if state.wrapper is None:
        return None, None, "Pipeline not loaded"
    if not video_path:
        return None, None, "No video"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, f"Cannot open: {video_path}"

    # Read current position or first frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None, "Cannot read frame"

    frame_resized = cv2.resize(frame, (state.width, state.height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Feed to ControlNet if enabled
    if state.wrapper.use_controlnet:
        state.wrapper.update_control_image(0, frame_pil)

    t0 = time.time()

    if state.wrapper.mode == "img2img":
        output = state.wrapper.img2img(frame_pil)
    else:
        output = state.wrapper.txt2img()

    dt = time.time() - t0

    # Convert output
    out_np = _output_to_numpy(output)

    return frame_rgb, out_np, f"Generated in {dt*1000:.0f}ms ({1/dt:.1f} FPS)"


def _output_to_numpy(output):
    """Convert wrapper output to numpy RGB array."""
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
            arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return arr
    return output


def stream_video(video_path, cn_scale, ipa_scale):
    """Generator that yields (input_frame, output_frame, status) for streaming."""
    if state.wrapper is None:
        yield None, None, "Pipeline not loaded"
        return
    if not video_path:
        yield None, None, "No video selected"
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, None, f"Cannot open: {video_path}"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state.running = True
    state.paused = False
    fps_smooth = 0.0
    frame_idx = 0

    try:
        while state.running:
            if state.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                # Loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx = 0

            t0 = time.time()

            frame_resized = cv2.resize(frame, (state.width, state.height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # ControlNet
            if state.wrapper.use_controlnet:
                state.wrapper.update_control_image(0, frame_pil)

            # IPAdapter from video frames (if no static style set)
            if state.wrapper.use_ipadapter:
                state.wrapper.update_style_image(frame_pil, is_stream=True)

            # Generate
            if state.wrapper.mode == "img2img":
                output = state.wrapper.img2img(frame_pil)
            else:
                output = state.wrapper.txt2img()

            out_np = _output_to_numpy(output)

            dt = time.time() - t0
            fps_instant = 1.0 / dt if dt > 0 else 0
            fps_smooth = fps_smooth * 0.9 + fps_instant * 0.1
            frame_idx += 1

            status = f"Frame {frame_idx}/{total_frames} | {fps_smooth:.1f} FPS | {dt*1000:.0f}ms/frame"

            yield frame_rgb, out_np, status

    finally:
        cap.release()
        state.running = False


def stop_stream():
    state.running = False
    return "Stopped"


def pause_stream():
    state.paused = not state.paused
    return "Paused" if state.paused else "Running"


# ============================================================
# UI
# ============================================================
def build_ui():
    with gr.Blocks(title="StreamDiffusion vid2vid") as app:
        gr.Markdown("## StreamDiffusion vid2vid")

        with gr.Row():
            # ---- LEFT: Controls ----
            with gr.Column(scale=1):
                # Model
                with gr.Accordion("Model", open=True):
                    model_preset = gr.Dropdown(
                        choices=list(MODEL_PRESETS.keys()),
                        value="sd-turbo",
                        label="Model Preset",
                    )
                    custom_model_id = gr.Textbox(
                        label="Custom Model ID (overrides preset)",
                        placeholder="e.g. runwayml/stable-diffusion-v1-5",
                    )
                    with gr.Row():
                        width = gr.Number(value=512, label="Width", precision=0)
                        height = gr.Number(value=512, label="Height", precision=0)
                    mode = gr.Radio(["img2img", "txt2img"], value="img2img", label="Mode")
                    acceleration = gr.Dropdown(ACCELERATION_TYPES, value="none", label="Acceleration")

                # Generation
                with gr.Accordion("Generation", open=True):
                    prompt = gr.Textbox(
                        value="masterpiece, high quality, detailed",
                        label="Prompt",
                        lines=2,
                    )
                    negative_prompt = gr.Textbox(
                        value="blurry, low quality, distorted",
                        label="Negative Prompt",
                    )
                    with gr.Row():
                        num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                        guidance_scale = gr.Slider(0.0, 20.0, value=1.0, step=0.1, label="Guidance Scale")
                    cfg_type = gr.Dropdown(CFG_TYPES, value="none", label="CFG Type")

                # ControlNet
                with gr.Accordion("ControlNet", open=False):
                    cn_preset = gr.Dropdown(
                        choices=list(CONTROLNET_PRESETS.keys()),
                        value="None",
                        label="ControlNet Preset",
                    )
                    cn_custom_model = gr.Textbox(
                        label="ControlNet Model ID (override)",
                        placeholder="e.g. lllyasviel/control_v11f1p_sd15_depth",
                    )
                    cn_preprocessor = gr.Dropdown(PREPROCESSORS, value="depth", label="Preprocessor")
                    cn_scale = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="ControlNet Scale")

                # IPAdapter
                with gr.Accordion("IPAdapter", open=False):
                    ipa_preset = gr.Dropdown(
                        choices=list(IPADAPTER_PRESETS.keys()),
                        value="None",
                        label="IPAdapter Preset",
                    )
                    ipa_scale = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="IPAdapter Scale")
                    style_image = gr.Image(label="Style / Reference Image", type="numpy")
                    update_style_btn = gr.Button("Update Style Image")

                # Load
                load_btn = gr.Button("Load Pipeline", variant="primary", size="lg")
                load_status = gr.Textbox(label="Status", interactive=False)

                # Prompt update (without reload)
                update_prompt_btn = gr.Button("Update Prompt (no reload)")

            # ---- RIGHT: Video + Output ----
            with gr.Column(scale=2):
                video_input = gr.Video(label="Input Video")

                with gr.Row():
                    stream_btn = gr.Button("Stream", variant="primary")
                    single_btn = gr.Button("Single Frame")
                    pause_btn = gr.Button("Pause/Resume")
                    stop_btn = gr.Button("Stop", variant="stop")

                stream_status = gr.Textbox(label="Stream Status", interactive=False)

                with gr.Row():
                    input_preview = gr.Image(label="Input Frame")
                    output_preview = gr.Image(label="Output")

        # ---- Wiring ----
        load_btn.click(
            fn=load_pipeline,
            inputs=[
                model_preset, custom_model_id,
                width, height,
                mode, acceleration,
                num_inference_steps, guidance_scale,
                cfg_type,
                prompt, negative_prompt,
                cn_preset, cn_custom_model, cn_preprocessor, cn_scale,
                ipa_preset, ipa_scale,
                style_image,
            ],
            outputs=[load_status],
        )

        update_prompt_btn.click(
            fn=update_prompt,
            inputs=[prompt, negative_prompt],
            outputs=[load_status],
        )

        update_style_btn.click(
            fn=update_style_image,
            inputs=[style_image],
            outputs=[load_status],
        )

        single_btn.click(
            fn=process_single_frame,
            inputs=[video_input],
            outputs=[input_preview, output_preview, stream_status],
        )

        stream_btn.click(
            fn=stream_video,
            inputs=[video_input, cn_scale, ipa_scale],
            outputs=[input_preview, output_preview, stream_status],
        )

        stop_btn.click(fn=stop_stream, outputs=[stream_status])
        pause_btn.click(fn=pause_stream, outputs=[stream_status])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
