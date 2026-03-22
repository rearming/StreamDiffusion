"""Standalone TRT engine rebuild for sdxl-turbo.
Loads model, builds engines, frees memory between steps.
"""
import os, sys, gc, torch

ENGINE_DIR = "X:/td/StreamDiffusion/engines/td"
MODEL_ID = "stabilityai/sdxl-turbo"
WIDTH, HEIGHT = 512, 512
MIN_BATCH, MAX_BATCH = 1, 4
MODE = "img2img"

os.makedirs(ENGINE_DIR, exist_ok=True)

print("=== Step 1: Load SDXL pipeline ===", flush=True)
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
single_file = hf_hub_download(MODEL_ID, "sd_xl_turbo_1.0_fp16.safetensors")
print(f"Using single file: {single_file}", flush=True)
pipe = StableDiffusionXLPipeline.from_single_file(single_file, torch_dtype=torch.float16)
# Only move UNet to GPU — needed for ONNX export tracing
pipe.unet = pipe.unet.to("cuda")
print(f"UNet on CUDA. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

print("\n=== Step 2: Create StreamDiffusion ===", flush=True)
sys.path.insert(0, "src")
from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import TorchVAEEncoder
from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType
from streamdiffusion.acceleration.tensorrt.models.models import UNet, VAE, VAEEncoder

stream = StreamDiffusion(
    pipe,
    t_index_list=[1],
    torch_dtype=torch.float16,
    width=WIDTH, height=HEIGHT,
    frame_buffer_size=1,
    use_denoising_batch=True,
    cfg_type="none",
)

# Detect model type
is_sdxl = hasattr(pipe, "text_encoder_2")
model_type = "SDXL" if is_sdxl else "SD15"
embedding_dim = 2048 if is_sdxl else 768
from streamdiffusion.model_detection import extract_unet_architecture, validate_architecture
unet_arch = extract_unet_architecture(stream.unet)
unet_arch = validate_architecture(unet_arch, model_type)
print(f"Model type: {model_type}, embedding_dim: {embedding_dim}", flush=True)

# Prepare engine paths
model_name = "sd_xl_turbo_1.0_fp16"
engine_subdir = f"{model_name}--lcm_lora-False--tiny_vae-True--min_batch-{MIN_BATCH}--max_batch-{MAX_BATCH}--mode-{MODE}"
full_engine_dir = os.path.join(ENGINE_DIR, engine_subdir)
os.makedirs(full_engine_dir, exist_ok=True)

from pathlib import Path
unet_path = Path(full_engine_dir) / "unet.engine"
vae_encoder_path = Path(full_engine_dir.replace(f"max_batch-{MAX_BATCH}", "max_batch-1")) / "vae_encoder.engine"
vae_decoder_path = Path(full_engine_dir.replace(f"max_batch-{MAX_BATCH}", "max_batch-1")) / "vae_decoder.engine"
os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)

engine_manager = EngineManager(full_engine_dir)

print(f"\nEngine dir: {full_engine_dir}", flush=True)
print(f"UNet path: {unet_path}", flush=True)
print(f"VAE enc path: {vae_encoder_path}", flush=True)
print(f"VAE dec path: {vae_decoder_path}", flush=True)

# Build options
build_opts = {
    'opt_image_height': HEIGHT,
    'opt_image_width': WIDTH,
    'build_dynamic_shape': True,
    'min_image_resolution': 384,
    'max_image_resolution': 1024,
}

from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export import UnifiedExportWrapper

# Use ControlNet support in UNet for compatibility
use_controlnet_trt = True

unet_model = UNet(
    fp16=True,
    device=torch.device("cuda"),
    max_batch_size=MAX_BATCH,
    min_batch_size=MIN_BATCH,
    embedding_dim=embedding_dim,
    unet_dim=stream.unet.config.in_channels,
    use_control=use_controlnet_trt,
    unet_arch=unet_arch if use_controlnet_trt else None,
    image_height=HEIGHT,
    image_width=WIDTH,
)

control_input_names = [name for name in unet_model.get_input_names() if name != 'ipadapter_scale']
wrapped_unet = UnifiedExportWrapper(
    stream.unet,
    use_controlnet=use_controlnet_trt,
    use_ipadapter=False,
    control_input_names=control_input_names,
    num_tokens=4
)

print("\n=== Step 3: Build VAE decoder ===", flush=True)
if not os.path.exists(vae_decoder_path):
    vae_decoder_model = VAE(device=torch.device("cuda"), max_batch_size=1, min_batch_size=1)
    engine_manager.compile_and_load_engine(
        EngineType.VAE_DECODER, vae_decoder_path,
        load_engine=False, model=stream.vae, model_config=vae_decoder_model,
        batch_size=1, cuda_stream=None, stream_vae=stream.vae,
        engine_build_options=build_opts,
    )
    print("VAE decoder built!", flush=True)
    gc.collect(); torch.cuda.empty_cache()
else:
    print("VAE decoder already exists, skipping", flush=True)

print("\n=== Step 4: Build VAE encoder ===", flush=True)
if not os.path.exists(vae_encoder_path):
    vae_encoder = TorchVAEEncoder(stream.vae)
    vae_encoder_model = VAEEncoder(device=torch.device("cuda"), max_batch_size=1, min_batch_size=1)
    engine_manager.compile_and_load_engine(
        EngineType.VAE_ENCODER, vae_encoder_path,
        load_engine=False, model=vae_encoder, model_config=vae_encoder_model,
        batch_size=1, cuda_stream=None,
        engine_build_options=build_opts,
    )
    print("VAE encoder built!", flush=True)
    gc.collect(); torch.cuda.empty_cache()
else:
    print("VAE encoder already exists, skipping", flush=True)

print("\n=== Step 5: Build UNet (this takes a while) ===", flush=True)
if not os.path.exists(unet_path):
    # Free everything except UNet from GPU before building
    stream.vae = stream.vae.to("cpu")
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.to("cpu")
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
    if hasattr(stream, 'text_encoder') and stream.text_encoder is not None:
        stream.text_encoder = stream.text_encoder.to("cpu")
    try:
        del vae_encoder
    except NameError:
        pass
    # Free ALL GPU memory — we don't need the model anymore, ONNX is cached
    del wrapped_unet, stream, pipe
    gc.collect(); torch.cuda.empty_cache()
    print(f"VRAM before UNet build: {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

    # Bypass compile_unet — ONNX already exists, just optimize + build engine
    from streamdiffusion.acceleration.tensorrt.utilities import optimize_onnx, build_engine
    onnx_path = str(unet_path) + ".onnx"
    onnx_opt_path = str(unet_path) + ".opt.onnx"

    print("Optimizing ONNX...", flush=True)
    optimize_onnx(onnx_path, onnx_opt_path, unet_model)

    print("Building TRT engine (this is the slow part)...", flush=True)
    unet_model.min_latent_shape = 384 // 8
    unet_model.max_latent_shape = 1024 // 8
    build_engine(str(unet_path), onnx_opt_path, unet_model, opt_batch_size=1, opt_image_height=HEIGHT, opt_image_width=WIDTH)
    print("UNet built!", flush=True)
else:
    print("UNet already exists, skipping", flush=True)

print("\n=== DONE! All engines built. ===", flush=True)
print(f"Engine dir: {full_engine_dir}", flush=True)
