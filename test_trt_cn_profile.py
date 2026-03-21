"""Profile TRT + ControlNet pipeline to find the ~920ms overhead."""
import os, sys, time, functools
os.environ['HF_HOME'] = 'X:/hf_cache'
sys.path.insert(0, 'X:/td/StreamDiffusion/src')

from PIL import Image
import numpy as np
import torch

# ── Timing utility ──
_timings = {}

def timed(label):
    """Decorator to time a function and accumulate stats."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if label not in _timings:
                _timings[label] = []
            _timings[label].append(dt * 1000)
            return result
        return wrapper
    return decorator

def print_timings():
    print("\n=== TIMING BREAKDOWN (avg ms over benchmark frames) ===")
    for label, times in sorted(_timings.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  {label:50s}  avg={avg:8.1f}ms  total={total:8.0f}ms  calls={len(times)}")
    print()

# ── Monkey-patch before creating wrapper ──
print("=== TRT + ControlNet PROFILING ===", flush=True)

from streamdiffusion.pipeline import StreamDiffusion
from streamdiffusion.modules.controlnet_module import ControlNetModule
from streamdiffusion.acceleration.tensorrt.runtime_engines.controlnet_engine import ControlNetModelEngine
from streamdiffusion.acceleration.tensorrt.utilities import Engine

# Patch StreamDiffusion.__call__ internals
_orig_sd_call = StreamDiffusion.__call__
def _profiled_sd_call(self, x=None):
    torch.cuda.synchronize()
    t_total = time.perf_counter()

    if x is not None:
        # preprocess
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = self.image_processor.preprocess(x, self.height, self.width).to(
            device=self.device, dtype=self.dtype
        )
        x = self._apply_image_preprocessing_hooks(x)
        torch.cuda.synchronize()
        _timings.setdefault("SD.__call__: preprocess", []).append((time.perf_counter() - t0) * 1000)

        if self.similar_image_filter:
            x = self.similar_filter(x)
            if x is None:
                time.sleep(self.inference_time_ema)
                return self.prev_image_result

        # encode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_t_latent = self.encode_image(x)
        torch.cuda.synchronize()
        _timings.setdefault("SD.__call__: encode_image (VAE enc)", []).append((time.perf_counter() - t0) * 1000)

        x_t_latent = self._apply_latent_preprocessing_hooks(x_t_latent)
    else:
        x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
            device=self.device, dtype=self.dtype
        )

    # predict_x0_batch (includes UNet + ControlNet hook)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_0_pred_out = self.predict_x0_batch(x_t_latent)
    torch.cuda.synchronize()
    _timings.setdefault("SD.__call__: predict_x0_batch (UNet+CN)", []).append((time.perf_counter() - t0) * 1000)

    x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)

    # clone latent
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    self.prev_latent_result = x_0_pred_out.detach().clone()
    torch.cuda.synchronize()
    _timings.setdefault("SD.__call__: latent clone", []).append((time.perf_counter() - t0) * 1000)

    # decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_output = self.decode_image(x_0_pred_out).detach().clone()
    torch.cuda.synchronize()
    _timings.setdefault("SD.__call__: decode_image (VAE dec)", []).append((time.perf_counter() - t0) * 1000)

    x_output = self._apply_image_postprocessing_hooks(x_output)
    self.prev_image_result = x_output

    torch.cuda.synchronize()
    _timings.setdefault("SD.__call__: TOTAL", []).append((time.perf_counter() - t_total) * 1000)

    return x_output

StreamDiffusion.__call__ = _profiled_sd_call

# Patch unet_step to time UNet vs hook separately
_orig_unet_step = StreamDiffusion.unet_step
def _profiled_unet_step(self, x_t_latent, t_list, idx=None):
    # Time the whole unet_step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_unet_step(self, x_t_latent, t_list, idx)
    torch.cuda.synchronize()
    _timings.setdefault("  unet_step: TOTAL", []).append((time.perf_counter() - t0) * 1000)
    return result
StreamDiffusion.unet_step = _profiled_unet_step

# Patch ControlNet hook execution (the build_unet_hook closure)
_orig_build_hook = ControlNetModule.build_unet_hook
def _profiled_build_hook(self):
    orig_hook = _orig_build_hook(self)
    def timed_hook(ctx):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = orig_hook(ctx)
        torch.cuda.synchronize()
        _timings.setdefault("    CN unet_hook: TOTAL", []).append((time.perf_counter() - t0) * 1000)
        return result
    return timed_hook
ControlNetModule.build_unet_hook = _profiled_build_hook

# Patch ControlNet TRT engine __call__
_orig_cn_engine_call = ControlNetModelEngine.__call__
def _profiled_cn_engine_call(self, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_cn_engine_call(self, **kwargs)
    torch.cuda.synchronize()
    _timings.setdefault("      CN TRT engine __call__", []).append((time.perf_counter() - t0) * 1000)
    return result
ControlNetModelEngine.__call__ = _profiled_cn_engine_call

# Patch Engine.allocate_buffers
_orig_alloc = Engine.allocate_buffers
def _profiled_alloc(self, shape_dict=None, device="cuda"):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_alloc(self, shape_dict, device)
    torch.cuda.synchronize()
    _timings.setdefault("      Engine.allocate_buffers", []).append((time.perf_counter() - t0) * 1000)
    return result
Engine.allocate_buffers = _profiled_alloc

# Patch Engine.infer
_orig_infer = Engine.infer
def _profiled_infer(self, feed_dict, stream, use_cuda_graph=False):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_infer(self, feed_dict, stream, use_cuda_graph)
    torch.cuda.synchronize()
    _timings.setdefault("      Engine.infer", []).append((time.perf_counter() - t0) * 1000)
    return result
Engine.infer = _profiled_infer

# Patch SDXL conditioning in controlnet_module
_orig_get_sdxl = ControlNetModule._get_cached_sdxl_conditioning
def _profiled_get_sdxl(self, ctx):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_get_sdxl(self, ctx)
    torch.cuda.synchronize()
    _timings.setdefault("      CN _get_cached_sdxl_cond", []).append((time.perf_counter() - t0) * 1000)
    return result
ControlNetModule._get_cached_sdxl_conditioning = _profiled_get_sdxl

# Patch prepare_frame_tensors
_orig_prepare_frame = ControlNetModule.prepare_frame_tensors
def _profiled_prepare_frame(self, device, dtype, batch_size):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_prepare_frame(self, device, dtype, batch_size)
    torch.cuda.synchronize()
    _timings.setdefault("      CN prepare_frame_tensors", []).append((time.perf_counter() - t0) * 1000)
    return result
ControlNetModule.prepare_frame_tensors = _profiled_prepare_frame

# Patch _check_unet_tensorrt
_orig_check_trt = StreamDiffusion._check_unet_tensorrt
def _profiled_check_trt(self):
    t0 = time.perf_counter()
    result = _orig_check_trt(self)
    _timings.setdefault("  _check_unet_tensorrt", []).append((time.perf_counter() - t0) * 1000)
    return result
StreamDiffusion._check_unet_tensorrt = _profiled_check_trt

# Patch _get_cached_sdxl_conditioning on pipeline
_orig_pipeline_sdxl = StreamDiffusion._get_cached_sdxl_conditioning
def _profiled_pipeline_sdxl(self, batch_size, cfg_type, guidance_scale):
    t0 = time.perf_counter()
    result = _orig_pipeline_sdxl(self, batch_size, cfg_type, guidance_scale)
    _timings.setdefault("  pipeline._get_cached_sdxl_cond", []).append((time.perf_counter() - t0) * 1000)
    return result
StreamDiffusion._get_cached_sdxl_conditioning = _profiled_pipeline_sdxl

# Patch _build_sdxl_conditioning on pipeline (cache miss path)
if hasattr(StreamDiffusion, '_build_sdxl_conditioning'):
    _orig_build_sdxl = StreamDiffusion._build_sdxl_conditioning
    def _profiled_build_sdxl(self, batch_size):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_build_sdxl(self, batch_size)
        torch.cuda.synchronize()
        _timings.setdefault("  pipeline._build_sdxl_cond (MISS)", []).append((time.perf_counter() - t0) * 1000)
        return result
    StreamDiffusion._build_sdxl_conditioning = _profiled_build_sdxl

# Patch wrapper preprocess_image
from streamdiffusion import StreamDiffusionWrapper
_orig_preprocess = StreamDiffusionWrapper.preprocess_image
def _profiled_preprocess(self, image):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_preprocess(self, image)
    torch.cuda.synchronize()
    _timings.setdefault("wrapper.preprocess_image", []).append((time.perf_counter() - t0) * 1000)
    return result
StreamDiffusionWrapper.preprocess_image = _profiled_preprocess

# Patch update_control_image
_orig_update_ctrl = StreamDiffusionWrapper.update_control_image
def _profiled_update_ctrl(self, index, image, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_update_ctrl(self, index, image, **kwargs)
    torch.cuda.synchronize()
    _timings.setdefault("wrapper.update_control_image", []).append((time.perf_counter() - t0) * 1000)
    return result
StreamDiffusionWrapper.update_control_image = _profiled_update_ctrl

# ── Now create the pipeline ──
print("[1] Loading pipeline...", flush=True)
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sdxl-turbo",
    t_index_list=[1],
    width=512, height=512,
    mode="img2img",
    frame_buffer_size=1,
    use_denoising_batch=True,
    use_lcm_lora=False,
    use_tiny_vae=True,
    acceleration="tensorrt",
    cfg_type="none",
    do_add_noise=True,
    warmup=5,
    seed=42,
    use_safety_checker=False,
    engine_dir="X:/td/StreamDiffusion/engines/td",
    use_controlnet=True,
    controlnet_config=[{
        "model_id": "xinsir/controlnet-depth-sdxl-1.0",
        "conditioning_scale": 0.5,
        "preprocessor": "canny",
        "preprocessor_params": {"low_threshold": 100, "high_threshold": 200},
        "enabled": True,
    }],
    use_ipadapter=False,
)

print("[2] Preparing...", flush=True)
wrapper.prepare(prompt="cyberpunk city neon", num_inference_steps=50, guidance_scale=1.0)

test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

print("[3] Warmup (5 frames, not timed)...", flush=True)
for i in range(5):
    wrapper.update_control_image(0, test_img)
    torch.cuda.synchronize()
    out = wrapper.img2img(test_img)
    torch.cuda.synchronize()
    print(f"  warmup {i} done", flush=True)

# Clear warmup timings
_timings.clear()

print("[4] Benchmark (10 frames, PROFILED)...", flush=True)
frame_times = []
for i in range(10):
    wrapper.update_control_image(0, test_img)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = wrapper.img2img(test_img)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    frame_times.append(dt)
    print(f"  frame {i}: {dt*1000:.0f}ms ({1/dt:.1f} FPS)", flush=True)

avg = sum(frame_times) / len(frame_times)
print(f"\n=== OVERALL: avg={avg*1000:.0f}ms ({1/avg:.1f} FPS) ===")

print_timings()
