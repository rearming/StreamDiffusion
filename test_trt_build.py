"""
Test TensorRT engine building with reduced memory limits for RTX 5080 (Blackwell).
Tries to build just a VAE decoder engine as a smoke test.
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"GPU mem free: {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB", flush=True)

# Check TRT version
import tensorrt as trt
print(f"TensorRT: {trt.__version__}", flush=True)

# Check RAM
import ctypes
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]
m = MEMORYSTATUSEX()
m.dwLength = ctypes.sizeof(m)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print(f"RAM: {m.ullAvailPhys/2**30:.1f} GiB free / {m.ullTotalPhys/2**30:.1f} GiB total", flush=True)
print(flush=True)

# Use the existing ONNX file if available
onnx_dir = "X:/td/StreamDiffusion/engines/td/stabilityai/sd-turbo--lcm_lora-True--tiny_vae-True--min_batch-1--max_batch-1--mode-img2img"
onnx_path = os.path.join(onnx_dir, "vae_decoder.engine.opt.onnx")
engine_path = os.path.join(onnx_dir, "vae_decoder_test.engine")

if not os.path.exists(onnx_path):
    print(f"ONNX file not found: {onnx_path}", flush=True)
    # Check alternative paths
    for root, dirs, files in os.walk("X:/td/StreamDiffusion/engines"):
        for f in files:
            if f.endswith(".onnx"):
                print(f"  Found: {os.path.join(root, f)}", flush=True)
    sys.exit(1)

print(f"Building TRT engine from: {onnx_path}", flush=True)
print(f"Output: {engine_path}", flush=True)
print(flush=True)

from streamdiffusion.acceleration.tensorrt.utilities import Engine

engine = Engine(engine_path)
t0 = time.time()
try:
    engine.build(
        onnx_path,
        fp16=True,
        input_profile={
            "latent": [(1, 4, 48, 48), (1, 4, 64, 64), (1, 4, 128, 128)]
        },
        enable_refit=False,
        enable_all_tactics=False,
        workspace_size=4 * 2**30,  # 4 GiB
    )
    t1 = time.time()
    print(f"\nSUCCESS! Engine built in {t1-t0:.1f}s", flush=True)
    print(f"Engine file: {engine_path} ({os.path.getsize(engine_path)/2**20:.1f} MiB)", flush=True)
except Exception as e:
    t1 = time.time()
    print(f"\nFAILED after {t1-t0:.1f}s: {e}", flush=True)
