#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util

from .models.models import CLIP, VAE, BaseModel, UNet, VAEEncoder

# Set up logger for this module
import logging
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

from ...model_detection import detect_model

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        
        # Buffer reuse optimization tracking
        self._last_shape_dict = None
        self._last_device = None

    def __del__(self):
        # Check if AttributeError: 'Engine' object has no attribute 'buffers'
        if not hasattr(self, 'buffers'):
            return
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        
        if hasattr(self, 'cuda_graph_instance') and self.cuda_graph_instance is not None:
            try:
                CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            except:
                pass
        if hasattr(self, 'graph') and self.graph is not None:
            try:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
            except:
                pass
        
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        logger.info(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                logger.warning(f"No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            logger.error("Failed to refit!")
            raise RuntimeError("TensorRT engine refit failed")

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        GiB = 2**30
        memory_limits = {}
        if workspace_size > 0:
            # Cap workspace at 4 GiB to reduce LLVM compilation memory pressure (RTX 50-series)
            capped = min(workspace_size, 4 * GiB)
            memory_limits[trt.MemoryPoolType.WORKSPACE] = capped
        # Limit tactic DRAM to 8 GiB to prevent LLVM OOM on Blackwell GPUs
        memory_limits[trt.MemoryPoolType.TACTIC_DRAM] = 8 * GiB
        if memory_limits:
            config_kwargs["memory_pool_limits"] = memory_limits
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(
                fp16=fp16, refittable=enable_refit, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Check if we can reuse existing buffers (OPTIMIZATION)
        if self._can_reuse_buffers(shape_dict, device):
            return
        
        # Clear existing buffers before reallocating
        self.tensors.clear()
        
        # Reset CUDA graph when buffers are reallocated
        # The captured graph becomes invalid with new memory addresses
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
            if hasattr(self, 'graph') and self.graph is not None:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
                self.graph = None
        
        # Two-pass allocation: set input shapes first, then allocate all tensors
        # This lets TensorRT infer output shapes from the inputs (needed for dynamic shapes)

        # Pass 1: set input shapes on the context
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if shape_dict and name in shape_dict:
                    self.context.set_input_shape(name, shape_dict[name])

        # Debug: check if all inputs were set
        _all_shapes_resolved = self.context.all_shape_inputs_specified
        if not _all_shapes_resolved:
            missing = []
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT and (not shape_dict or name not in shape_dict):
                    missing.append(name)
            print(f"[TRT allocate_buffers] NOT all input shapes set! missing={missing}", flush=True)
            print(f"[TRT allocate_buffers] shape_dict keys={list(shape_dict.keys()) if shape_dict else 'None'}", flush=True)

        # Pass 2: allocate tensors (output shapes are now resolved by the context)
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)

            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            elif mode == trt.TensorIOMode.OUTPUT:
                # Get inferred output shape from context (resolves dynamic dims)
                shape = self.context.get_tensor_shape(name)
                if any(d < 0 for d in shape):
                    print(f"[TRT] OUTPUT '{name}' still has dynamic dims: {list(shape)}", flush=True)
                    print(f"[TRT] all_inputs_specified={self.context.all_shape_inputs_specified}", flush=True)
                    # Fallback: match output to 'sample' input shape if available
                    if shape_dict and 'sample' in shape_dict:
                        shape = shape_dict['sample']
                        print(f"[TRT] Using sample shape as fallback: {list(shape)}", flush=True)
            else:
                shape = self.engine.get_tensor_shape(name)

            dtype_np = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            tensor = torch.empty(tuple(shape),
                                 dtype=numpy_to_torch_dtype_dict[dtype_np]) \
                          .to(device=device)
            self.tensors[name] = tensor
        
        # Cache allocation parameters for reuse check
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device
    
    def _can_reuse_buffers(self, shape_dict=None, device="cuda"):
        """
        Check if existing buffers can be reused (avoiding expensive reallocation)
        
        Returns:
            bool: True if buffers can be reused, False if reallocation needed
        """
        # No existing tensors - need to allocate
        if not self.tensors:
            return False
        
        # Device changed - need to reallocate
        if not hasattr(self, '_last_device') or self._last_device != device:
            return False
        
        # No cached shape_dict - need to allocate
        if not hasattr(self, '_last_shape_dict'):
            return False
            
        # Compare current vs cached shape_dict
        if shape_dict is None and self._last_shape_dict is None:
            return True
        elif shape_dict is None or self._last_shape_dict is None:
            return False
        
        # Quick check: if tensor counts differ, can't reuse
        if len(shape_dict) != len(self._last_shape_dict):
            return False
        
        # Compare shapes for all tensors in the new shape_dict
        for name, new_shape in shape_dict.items():
            # Check if tensor exists in cached shapes
            cached_shape = self._last_shape_dict.get(name)
            if cached_shape is None:
                return False
            
            # Compare shapes (handle different types consistently)
            if tuple(cached_shape) != tuple(new_shape):
                return False
        
        return True

    def reset_cuda_graph(self):
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
        if hasattr(self, 'graph') and self.graph is not None:
            CUASSERT(cudart.cudaGraphDestroy(self.graph))
            self.graph = None

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
):
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        activation_carveout = 4 * GiB
        max_workspace_size = free_mem - activation_carveout
    else:
        max_workspace_size = 0
    engine = Engine(engine_path)
    input_profile = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )
    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=input_profile,
        enable_refit=build_enable_refit,
        enable_all_tactics=build_all_tactics,
        workspace_size=max_workspace_size,
    )

    return engine





def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    # TODO: Not 100% happy about this function - needs refactoring
    
    is_sdxl = False
    is_sdxl_controlnet = False

    # Detect if this is a ControlNet model (vs UNet model)
    is_controlnet = (
        hasattr(model, '__class__') and 'ControlNet' in model.__class__.__name__
    ) or (
        hasattr(model, 'config') and hasattr(model.config, '_class_name') and
        'ControlNet' in model.config._class_name
    )

    # Detect if this is an SDXL model via detect_model
    if hasattr(model, 'unet'):
        detection_result = detect_model(model.unet)
        if detection_result is not None:
            is_sdxl = detection_result.get('is_sdxl', False)
    elif hasattr(model, 'config'):
        detection_result = detect_model(model)
        if detection_result is not None:
            is_sdxl = detection_result.get('is_sdxl', False)
    
    # Detect if this is an SDXL ControlNet
    is_sdxl_controlnet = is_controlnet and (is_sdxl or (
        hasattr(model, 'config') and
        getattr(model.config, 'addition_embed_type', None) == 'text_time'
    ))
    
    wrapped_model = model  # Default: use model as-is
    
    is_vae = isinstance(model_data, (VAE, VAEEncoder))
    
    if is_vae:
        logger.info(f"Exporting {model_data.name} (no SDXL wrapping needed)...")
    elif is_sdxl and not is_controlnet:
        embedding_dim = getattr(model_data, 'embedding_dim', 'unknown')
        logger.info(f"Detected SDXL model (embedding_dim={embedding_dim}), using wrapper for ONNX export...")
        from .export_wrappers.unet_sdxl_export import SDXLExportWrapper
        wrapped_model = SDXLExportWrapper(model)
    elif not is_controlnet:
        embedding_dim = getattr(model_data, 'embedding_dim', 'unknown')
        logger.info(f"Detected non-SDXL model (embedding_dim={embedding_dim}), using model as-is for ONNX export...")
    
    # SDXL ControlNet models need special wrapper for added_cond_kwargs
    elif is_sdxl_controlnet:
        logger.info("Detected SDXL ControlNet model, using specialized wrapper...")
        from .export_wrappers.controlnet_export import SDXLControlNetExportWrapper
        wrapped_model = SDXLControlNetExportWrapper(model)
    
    # Regular ControlNet models are exported directly
    elif is_controlnet:
        logger.info("Detected ControlNet model, exporting directly...")
        wrapped_model = model
    
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        
        # Determine if we need external data format for large models (like SDXL)
        is_large_model = is_sdxl or (hasattr(model, 'config') and getattr(model.config, 'sample_size', 32) >= 64)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            torch.onnx.export(
                wrapped_model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=model_data.get_input_names(),
                output_names=model_data.get_output_names(),
                dynamic_axes=model_data.get_dynamic_axes(),
            )
        
        # Convert to external data format for large models (SDXL)
        if is_large_model:
            import os
            
            # Load the exported model
            onnx_model = onnx.load(onnx_path)
            
            # Check if model is large enough to need external data
            if onnx_model.ByteSize() > 2147483648:  # 2GB
                # Create directory for external data
                onnx_dir = os.path.dirname(onnx_path)
                
                # Re-save with external data format
                onnx.save_model(
                    onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
                logger.info(f"Converted to external data format with weights in weights.pb")
            
            del onnx_model
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    import os
    import shutil
    
    # Check if external data files exist (indicating external data format was used)
    onnx_dir = os.path.dirname(onnx_path)
    onnx_basename = os.path.basename(onnx_path)
    # External data can be .pb files or extensionless files (e.g. onnx__Add_18237)
    external_data_files = [
        f for f in os.listdir(onnx_dir)
        if os.path.isfile(os.path.join(onnx_dir, f))
        and f != onnx_basename
        and (f.endswith('.pb') or '.' not in f)
    ]
    uses_external_data = len(external_data_files) > 0
    
    if uses_external_data:
        # Load model with external data
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx_opt_graph = model_data.optimize(onnx_model)
        
        # Create output directory
        opt_dir = os.path.dirname(onnx_opt_path)
        os.makedirs(opt_dir, exist_ok=True)
        
        # Clean up existing files in output directory
        if os.path.exists(opt_dir):
            for f in os.listdir(opt_dir):
                if f.endswith('.pb') or f.endswith('.onnx'):
                    os.remove(os.path.join(opt_dir, f))
        
        # Save optimized model with external data format
        onnx.save_model(
            onnx_opt_graph,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        logger.info(f"ONNX optimization complete with external data")
        
    else:
        # Check total size including external data files in the same directory
        source_size = os.path.getsize(onnx_path)
        total_size = sum(
            os.path.getsize(os.path.join(onnx_dir, f))
            for f in os.listdir(onnx_dir)
        )
        is_large_model = total_size > 512 * 1024 * 1024  # >512MB total

        if is_large_model:
            logger.info(f"Large ONNX model (total {total_size / (1024**3):.2f} GB), skipping graphsurgeon optimization")
            logger.info("TensorRT will optimize during engine build.")
            # Copy ONNX and all external data files to opt path directory
            import shutil
            opt_dir = os.path.dirname(onnx_opt_path)
            os.makedirs(opt_dir, exist_ok=True)
            # Skip copy if src and dst resolve to the same file
            if os.path.abspath(onnx_path) != os.path.abspath(onnx_opt_path):
                shutil.copy2(onnx_path, onnx_opt_path)
            for f in os.listdir(onnx_dir):
                src = os.path.join(onnx_dir, f)
                dst = os.path.join(opt_dir, f)
                if os.path.abspath(src) != os.path.abspath(dst) and os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
            onnx.save(onnx_opt_graph, onnx_opt_path)
    
    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
