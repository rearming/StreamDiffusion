# TRT ControlNet VRAM Pressure Fix

## The Problem

With the full SDXL-turbo TensorRT pipeline loaded (UNet + VAE + ControlNet engines),
the ControlNet TRT engine ran at **~900ms per call** instead of the expected **~8ms**.
This made the integrated pipeline run at **1 FPS** despite each engine being fast standalone.

## Root Cause: GPU Memory Pressure

The RTX 5080 has 16GB VRAM. After loading everything, memory usage was:

| Component             | VRAM     |
|-----------------------|----------|
| TRT UNet engine       | ~2.5 GB  |
| TRT ControlNet engine | ~2.4 GB  |
| TRT VAE engines       | ~0.2 GB  |
| SDXL CLIP text encoder| ~5.2 GB  |
| Scheduler/buffers/etc | ~1.5 GB  |
| **Total**             | **~13.8 GB** |
| **Free**              | **~2.2 GB**  |

The CLIP text encoder was sitting idle on GPU after prompt encoding — it's only needed
when encoding a new prompt (during `prepare()` or `update_prompt()`), not during inference.

With only 2.2 GB free, the TRT ControlNet engine couldn't allocate proper workspace memory
for its inference kernels. TensorRT silently fell back to much slower execution paths,
causing the 100x slowdown (8ms → 900ms).

### How We Found It

1. **Profiling test** (`test_trt_cn_profile.py`) showed all 900ms was inside
   `ControlNetModelEngine.__call__`.

2. **Per-stage timing** inside `__call__` revealed that `execute_async_v3` launched in 3ms
   (CPU-side), but `torch.cuda.synchronize()` after it took 900ms — meaning the GPU
   kernels genuinely took that long.

3. **Standalone test** (`test_cn_engine_standalone.py`) confirmed the CN engine runs at
   8ms when loaded alone (plenty of VRAM).

4. **Memory analysis** (`test_cn_mem_check.py`) showed 13.8GB/16.3GB used. Moving the
   text encoder to CPU freed 5.2GB and immediately fixed the speed.

## The Fix

In `wrapper.py`, text encoders are now automatically offloaded to CPU after prompt
encoding and reloaded on-demand:

- `_offload_text_encoders()` — called after `__init__`'s initial `stream.prepare()`
  and after every `prepare()` / `update_prompt()` / `update_stream_params()` call
- `_reload_text_encoders()` — called before any prompt encoding operation

This is transparent to callers. Prompt changes take a small extra hit (~200ms) to move
the encoder back to GPU, but inference runs unimpeded.

### VRAM After Fix

| Component             | VRAM     |
|-----------------------|----------|
| TRT UNet engine       | ~2.5 GB  |
| TRT ControlNet engine | ~2.4 GB  |
| TRT VAE engines       | ~0.2 GB  |
| Scheduler/buffers/etc | ~1.5 GB  |
| **Total**             | **~8.4 GB** |
| **Free**              | **~7.6 GB**  |

## Results

| Configuration     | Before Fix | After Fix |
|-------------------|-----------|-----------|
| UNet only         | 35 FPS    | **38 FPS** |
| UNet + ControlNet | **1 FPS** | **27 FPS** |
| CN engine latency | 900 ms    | **8 ms**   |

## ControlNet Preprocessor Note

The ControlNet **model** and **preprocessor** are separate:
- **Preprocessor**: converts input frame to guidance image (canny=edges, depth=depth map)
- **Model**: neural net trained on a specific type of guidance image

They should match (depth model + depth preprocessor), but the depth preprocessor runs a
full neural network (MiDaS) on GPU and cuts FPS roughly in half (~12 FPS vs ~22 FPS).

The canny preprocessor is just OpenCV edge detection (CPU, near-zero cost). Using
`canny preprocessor + depth model` is technically a mismatch but produces usable results
at full speed. For proper matching without the FPS hit, swap to a canny ControlNet model
(`diffusers/controlnet-canny-sdxl-1.0`) and rebuild the TRT engine.

## Lesson

On 16GB GPUs running SDXL with TRT, VRAM is tight. Any unused model left on GPU
can starve TRT engines of workspace memory, causing silent catastrophic slowdowns.
Always offload models that aren't needed during the inference hot path.
