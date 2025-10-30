from .base import BasePreprocessor, PipelineAwareProcessor
from typing import Any
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor
from .openpose import OpenPosePreprocessor
from .lineart import LineartPreprocessor
from .standard_lineart import StandardLineartPreprocessor
from .passthrough import PassthroughPreprocessor
from .external import ExternalPreprocessor
from .soft_edge import SoftEdgePreprocessor
from .hed import HEDPreprocessor
from .ipadapter_embedding import IPAdapterEmbeddingPreprocessor
from .faceid_embedding import FaceIDEmbeddingPreprocessor
from .feedback import FeedbackPreprocessor
from .latent_feedback import LatentFeedbackPreprocessor
from .sharpen import SharpenPreprocessor
from .upscale import UpscalePreprocessor
from .blur import BlurPreprocessor
from .realesrgan_trt import RealESRGANProcessor

# Try to import TensorRT preprocessors - might not be available on all systems
try:
    from .depth_tensorrt import DepthAnythingTensorrtPreprocessor
    DEPTH_TENSORRT_AVAILABLE = True
except ImportError:
    DepthAnythingTensorrtPreprocessor = None
    DEPTH_TENSORRT_AVAILABLE = False

try:
    from .pose_tensorrt import YoloNasPoseTensorrtPreprocessor
    POSE_TENSORRT_AVAILABLE = True
except ImportError:
    YoloNasPoseTensorrtPreprocessor = None
    POSE_TENSORRT_AVAILABLE = False

try:
    from .temporal_net_tensorrt import TemporalNetTensorRTPreprocessor
    TEMPORAL_NET_TENSORRT_AVAILABLE = True
except ImportError:
    TemporalNetTensorRTPreprocessor = None
    TEMPORAL_NET_TENSORRT_AVAILABLE = False

try:
    from .mediapipe_pose import MediaPipePosePreprocessor
    MEDIAPIPE_POSE_AVAILABLE = True
except ImportError:
    MediaPipePosePreprocessor = None
    MEDIAPIPE_POSE_AVAILABLE = False

try:
    from .mediapipe_segmentation import MediaPipeSegmentationPreprocessor
    MEDIAPIPE_SEGMENTATION_AVAILABLE = True
except ImportError:
    MediaPipeSegmentationPreprocessor = None
    MEDIAPIPE_SEGMENTATION_AVAILABLE = False

# Registry for easy lookup
_preprocessor_registry = {
    "canny": CannyPreprocessor,
    "depth": DepthPreprocessor,
    "openpose": OpenPosePreprocessor,
    "lineart": LineartPreprocessor,
    "standard_lineart": StandardLineartPreprocessor,
    "passthrough": PassthroughPreprocessor,
    "external": ExternalPreprocessor,
    "soft_edge": SoftEdgePreprocessor,
    "hed": HEDPreprocessor,
    "feedback": FeedbackPreprocessor,
    "latent_feedback": LatentFeedbackPreprocessor,
    "sharpen": SharpenPreprocessor,
    "upscale": UpscalePreprocessor,
    "blur": BlurPreprocessor,
    "realesrgan_trt": RealESRGANProcessor,
}   

# Add TensorRT preprocessors if available
if DEPTH_TENSORRT_AVAILABLE:
    _preprocessor_registry["depth_tensorrt"] = DepthAnythingTensorrtPreprocessor

if POSE_TENSORRT_AVAILABLE:
    _preprocessor_registry["pose_tensorrt"] = YoloNasPoseTensorrtPreprocessor

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    _preprocessor_registry["temporal_net_tensorrt"] = TemporalNetTensorRTPreprocessor

# Add MediaPipe preprocessors if available
if MEDIAPIPE_POSE_AVAILABLE:
    _preprocessor_registry["mediapipe_pose"] = MediaPipePosePreprocessor

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    _preprocessor_registry["mediapipe_segmentation"] = MediaPipeSegmentationPreprocessor


def get_preprocessor_class(name: str) -> type:
    """
    Get a preprocessor class by name
    
    Args:
        name: Name of the preprocessor
        
    Returns:
        Preprocessor class
        
    Raises:
        ValueError: If preprocessor name is not found
    """
    if name not in _preprocessor_registry:
        available = ", ".join(_preprocessor_registry.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")
    
    return _preprocessor_registry[name]


def get_preprocessor(name: str, pipeline_ref: Any = None) -> BasePreprocessor:
    """
    Get a preprocessor by name
    
    Args:
        name: Name of the preprocessor
        pipeline_ref: Pipeline reference for pipeline-aware processors (required for some processors)
        
    Returns:
        Preprocessor instance
        
    Raises:
        ValueError: If preprocessor name is not found or pipeline_ref missing for pipeline-aware processor
    """
    processor_class = get_preprocessor_class(name)
    
    # Check if this is a pipeline-aware processor
    if hasattr(processor_class, 'requires_sync_processing') and processor_class.requires_sync_processing:
        if pipeline_ref is None:
            raise ValueError(f"Processor '{name}' requires a pipeline_ref")
        return processor_class(pipeline_ref=pipeline_ref, _registry_name=name)
    else:
        return processor_class(_registry_name=name)


def register_preprocessor(name: str, preprocessor_class):
    """
    Register a new preprocessor
    
    Args:
        name: Name to register under
        preprocessor_class: Preprocessor class
    """
    _preprocessor_registry[name] = preprocessor_class


def list_preprocessors():
    """List all available preprocessors"""
    return list(_preprocessor_registry.keys())


__all__ = [
    "BasePreprocessor",
    "PipelineAwareProcessor",
    "CannyPreprocessor",
    "DepthPreprocessor", 
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "StandardLineartPreprocessor",
    "PassthroughPreprocessor",
    "ExternalPreprocessor",
    "SoftEdgePreprocessor",
    "HEDPreprocessor",
    "IPAdapterEmbeddingPreprocessor",
    "FaceIDEmbeddingPreprocessor",
    "FeedbackPreprocessor",
    "LatentFeedbackPreprocessor",
    "get_preprocessor",
    "get_preprocessor_class",
    "register_preprocessor",
    "list_preprocessors",
]

if DEPTH_TENSORRT_AVAILABLE:
    __all__.append("DepthAnythingTensorrtPreprocessor")

if POSE_TENSORRT_AVAILABLE:
    __all__.append("YoloNasPoseTensorrtPreprocessor")

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    __all__.append("TemporalNetTensorRTPreprocessor")

if MEDIAPIPE_POSE_AVAILABLE:
    __all__.append("MediaPipePosePreprocessor")

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    __all__.append("MediaPipeSegmentationPreprocessor")


# =============================================================================
# CUSTOM PROCESSOR DISCOVERY
# =============================================================================

import logging
import os
logger = logging.getLogger(__name__)

def _discover_custom_processors():
    """
    Auto-discover custom processors from custom_processors/ folder.

    Supports:
    1. Processor collection repos with processors.yaml manifest
    2. Standalone processor folders (fallback to auto-scan .py files)

    Convention:
    - Clone repos to custom_processors/<name>/
    - Use absolute imports: from streamdiffusion.preprocessing.processors import BasePreprocessor
    - Folder name becomes namespace for processors
    """
    import importlib.util
    import inspect
    from pathlib import Path

    # Allow disabling in production/Docker
    if os.getenv("STREAMDIFFUSION_DISABLE_CUSTOM_PROCESSORS") == "1":
        logger.info("Custom processor discovery disabled via environment variable")
        return

    try:
        # Navigate to repo_root/custom_processors/
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        custom_dir = repo_root / "custom_processors"

        if not custom_dir.exists():
            logger.debug("custom_processors/ folder not found, skipping discovery")
            return

        logger.info("Scanning custom_processors/ folder for custom processors...")

        # Scan for processor collections/folders
        for item in custom_dir.iterdir():
            if not item.is_dir() or item.name.startswith(('.', '_')):
                continue

            # Check for processors.yaml manifest
            manifest_file = item / "processors.yaml"

            if manifest_file.exists():
                _load_processor_collection(item, manifest_file)
            else:
                # Fallback: auto-scan for .py files
                _load_processor_folder_auto(item)

    except Exception as e:
        logger.error(f"Custom processor discovery failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _load_processor_collection(collection_dir, manifest_file):
    """Load processors from a collection with processors.yaml manifest"""
    import yaml

    try:
        with open(manifest_file, 'r') as f:
            manifest = yaml.safe_load(f)

        collection_name = collection_dir.name
        processor_files = manifest.get('processors', [])

        if not processor_files:
            logger.warning(f"Collection '{collection_name}' has empty processors list")
            return

        logger.info(f"Loading processor collection '{collection_name}' ({len(processor_files)} processors)")

        for proc_file in processor_files:
            if isinstance(proc_file, dict):
                # Extended format with metadata
                filename = proc_file.get('file')
                enabled = proc_file.get('enabled', True)
                if not enabled:
                    logger.info(f"  Skipping disabled processor: {filename}")
                    continue
            else:
                # Simple format: just filename
                filename = proc_file

            proc_path = collection_dir / filename

            if not proc_path.exists():
                logger.warning(f"  Processor file not found: {filename}")
                continue

            # Extract processor name from filename (strip .py)
            proc_name = proc_path.stem

            _load_processor_from_file(proc_path, proc_name)

    except Exception as e:
        logger.error(f"Failed to load collection from {collection_dir.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _load_processor_folder_auto(folder):
    """Auto-discover processors by scanning for .py files (no manifest)"""
    logger.info(f"Auto-scanning folder for processors: {folder.name}")

    found_any = False
    for py_file in folder.glob("*.py"):
        # Skip special files
        if py_file.name.startswith('_') or py_file.name in ['base.py', 'setup.py']:
            continue

        proc_name = py_file.stem
        _load_processor_from_file(py_file, proc_name)
        found_any = True

    if not found_any:
        logger.warning(f"  No processor files found in {folder.name}")


def _load_processor_from_file(file_path, proc_name):
    """Load a processor class from a Python file"""
    import importlib.util
    import inspect

    try:
        spec = importlib.util.spec_from_file_location(
            f"custom_processors.{file_path.parent.name}.{file_path.stem}",
            file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find processor class
        found_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a valid processor (but not base classes)
            if (issubclass(obj, (BasePreprocessor, PipelineAwareProcessor)) and
                obj not in [BasePreprocessor, PipelineAwareProcessor]):
                found_classes.append((name, obj))

        if len(found_classes) == 0:
            logger.warning(f"  No valid processor class found in {file_path.name}")
            return

        if len(found_classes) > 1:
            logger.warning(f"  Multiple processor classes found in {file_path.name}, using first: {found_classes[0][0]}")

        processor_class = found_classes[0][1]

        # Check for name collision with core processors
        if proc_name in _preprocessor_registry:
            logger.error(f"  Name conflict: processor '{proc_name}' already exists in registry, skipping")
            return

        # Register the processor
        register_preprocessor(proc_name, processor_class)
        logger.info(f"  Registered custom processor: {proc_name} ({processor_class.__name__})")

    except Exception as e:
        logger.error(f"  Failed to load {file_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

_discover_custom_processors()