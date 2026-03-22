# SDXL TensorRT Engine Rebuild — Incident Report (2026-03-22)

## Summary

Full debug session to fix SDXL-turbo TRT engine loading/building on RTX 5080 (16GB VRAM, 32GB RAM, Windows 10). Started with a HuggingFace 401 error, cascaded into multiple segfaults and OOM issues during engine rebuild.

## System Configuration

- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM, Blackwell SM 12.x)
- **RAM**: 32GB + 16GB page file
- **OS**: Windows 10 Enterprise
- **Python**: 3.11
- **PyTorch**: 2.7.0+cu128
- **TensorRT**: 10.15.1.29
- **diffusers**: 0.30.3
- **safetensors**: 0.7.0
- **ONNX**: 1.16.1

## Model: stabilityai/sdxl-turbo

- Loaded via `from_single_file` using `sd_xl_turbo_1.0_fp16.safetensors` (6.9GB)
- VRAM footprint: ~6.7GB full pipeline, ~5GB UNet only
- TRT UNet engine: ~5.1GB

## Chain of Errors & Fixes

### 1. HuggingFace 401 on `list_repo_tree`

**Error**: `list_repo_tree("stabilityai/sdxl-turbo")` returned 401 Unauthorized.

**Root cause**: No HF token configured (`HfFolder.get_token()` returned None).

**Fix**: `huggingface-cli login` + graceful fallback in `wrapper.py` — catch `list_repo_tree` failure and fall through to `from_pretrained` which handles auth failures internally.

**File**: `src/streamdiffusion/wrapper.py` — wrapped `list_repo_tree` in try/except.

### 2. `UnboundLocalError: onnx_opt_graph`

**Error**: `del onnx_opt_graph` in `optimize_onnx()` when the variable was never assigned (large model path skips optimization).

**Fix**: Wrapped in `try/except NameError`.

**File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py`

### 3. Segfault loading ControlNet PyTorch model

**Error**: Process segfault during `ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")`.

**Root cause**: Windows `safetensors` mmap fails when system virtual memory is exhausted — TRT engines + SDXL model consume too much. The `os error 1455` ("paging file too small") manifests as a segfault.

**Fix**: `add_controlnet_lightweight()` — skip PyTorch ControlNet model loading when a pre-built TRT engine exists. Register preprocessor/config only.

**File**: `src/streamdiffusion/modules/controlnet_module.py`, `src/streamdiffusion/wrapper.py`

### 4. fp32 UNet safetensors segfault

**Error**: `safetensors.load_file` segfaults on the 10GB fp32 UNet file.

**Root cause**: HF cache had the fp32 variant (10GB) of `diffusion_pytorch_model.safetensors`. Loading 10GB via mmap on Windows with active CUDA context causes segfault.

**Fix**: Pass `variant="fp16"` to `from_pretrained` so diffusers loads the 5GB fp16 weights instead. Fallback to no-variant if fp16 doesn't exist.

**File**: `src/streamdiffusion/wrapper.py`

### 5. SDXL ONNX export segfault (double-wrap bug)

**Error**: `torch.onnx.export` segfaults during JIT tracing.

**Root cause**: `export_onnx()` in `utilities.py` wraps the model with `SDXLExportWrapper`, but the model was ALREADY wrapped with `UnifiedExportWrapper` (from the build pipeline). The `SDXLExportWrapper.__init__` runs a test forward pass through the double-wrapped model with wrong inputs (missing ControlNet tensors), causing a segfault in JIT internals.

**Fix**: Skip the test forward pass when `SDXLExportWrapper` detects it's wrapping a pre-wrapped model (check `base_unet != unet`). Detect SDXL support from `config.addition_embed_type` instead.

**File**: `src/streamdiffusion/acceleration/tensorrt/export_wrappers/unet_sdxl_export.py`

### 6. ONNX external data — thousands of files

**Error**: `torch.onnx.export` with `torch.autocast("cuda")` exports weights as 1124 individual external data files instead of embedding them in the ONNX protobuf.

**Root cause**: PyTorch 2.7's ONNX exporter with autocast creates per-tensor external data files. Loading these back overwhelms the OS.

**Fix**: After export, consolidate with `onnx.save_model(..., save_as_external_data=True, all_tensors_to_one_file=True, location="weights.pb")`. Also added file size check before `onnx.load()` to avoid loading the full model unnecessarily.

**File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py`, `rebuild_engines.py`

### 7. graphsurgeon protobuf error on large ONNX

**Error**: `google.protobuf.message.DecodeError: Error parsing message` during `optimize_onnx`.

**Root cause**: `optimize_onnx` tried to run graphsurgeon on a 5GB ONNX model with external data. Protobuf can't handle models this large in-memory.

**Fix**: Added size check (`> 512MB`) in the `uses_external_data` branch of `optimize_onnx` — skip graphsurgeon and let TensorRT optimize during engine build instead.

**File**: `src/streamdiffusion/acceleration/tensorrt/utilities.py`

### 8. CUDA OOM loading TRT engine

**Error**: `OutOfMemory: Requested size was 5134951808 bytes` when deserializing UNet TRT engine.

**Root cause**: Full SDXL pipeline (~6.7GB) still on CUDA when TRT engine (~5GB) tries to load. 6.7 + 5 > 16GB.

**Fix**: Offload PyTorch UNet + text encoders to CPU BEFORE loading TRT engine. After TRT loads and PyTorch UNet/VAE are deleted, restore text encoders to CUDA for prompt encoding.

**File**: `src/streamdiffusion/wrapper.py`

### 9. CUDA memory fragmentation

**Error**: `CUDA out of memory. Tried to allocate 30.00 MiB` with 7.84 GiB free.

**Root cause**: CUDA memory fragmentation from failed loading attempts leaving partial allocations.

**Fix**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before any torch import. Also clear `torch.cuda.empty_cache()` between loading attempts and move pipeline components to CUDA individually instead of `pipe.to(device)` all at once.

**File**: `gradio_vid2vid.py`, `src/streamdiffusion/wrapper.py`

### 10. Text encoders on wrong device after TRT load

**Error**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`

**Root cause**: Text encoders were offloaded to CPU for TRT engine loading but never moved back before `stream.prepare()` tried to encode prompts.

**Fix**: Restore text encoders to CUDA after TRT engines load and PyTorch UNet/VAE are freed.

**File**: `src/streamdiffusion/wrapper.py`

## Files Modified

| File | Changes |
|------|---------|
| `gradio_vid2vid.py` | `PYTORCH_CUDA_ALLOC_CONF` env var |
| `src/streamdiffusion/wrapper.py` | HF 401 fallback, fp16 variant, VRAM offloading for TRT load, text encoder restore, component-wise CUDA move, `add_controlnet_lightweight` integration |
| `src/streamdiffusion/modules/controlnet_module.py` | `add_controlnet_lightweight()` method, `_setup_preprocessor()` refactor |
| `src/streamdiffusion/acceleration/tensorrt/utilities.py` | `onnx_opt_graph` fix, `onnx.load` size check, external data large model bypass, debug prints |
| `src/streamdiffusion/acceleration/tensorrt/export_wrappers/unet_sdxl_export.py` | Skip test forward pass for pre-wrapped models |
| `src/streamdiffusion/acceleration/tensorrt/export_wrappers/unet_controlnet_export.py` | Remove emoji from debug print (cp1251 encoding crash) |
| `rebuild_engines.py` | Standalone TRT engine rebuild script |

## Rebuild Procedure

If engines need rebuilding in the future:

```bash
# 1. Ensure HF auth
huggingface-cli login

# 2. Run standalone rebuild (handles memory carefully)
python rebuild_engines.py

# 3. If ONNX export segfaults, check for corrupted .onnx files (should be >1GB, not 5MB)
# 4. After ONNX export, consolidate external data if needed:
#    python -c "import onnx; m = onnx.load('path.onnx', load_external_data=True); onnx.save_model(m, 'path.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='weights.pb')"
```

## Known Remaining Issues

- **ControlNet output quality**: CN scale produces messy output instead of structured patterns. May be related to `add_controlnet_lightweight` skipping PyTorch model or TRT engine input mismatch. Needs investigation.
- **Blackwell TRT ControlNet compilation**: Skipped on SM >= 12 (causes segfault). Pre-built engines required.
