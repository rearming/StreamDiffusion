# Custom Processors

This folder contains custom processor collections that extend StreamDiffusion's preprocessing capabilities.

## Quick Start

1. Clone a processor collection repository into this folder:
   ```bash
   git clone https://github.com/your-repo/my-processors custom_processors/my-processors
   ```

2. The processors will be automatically discovered and registered when StreamDiffusion starts.

3. In TouchDesigner, pulse the `Refreshfxmetadata` parameter to load the new processors.

## Folder Structure

```
custom_processors/
├── README.md (this file)
├── csp/                          # Spencer's CSP processor collection
│   ├── processors.yaml           # Manifest (optional)
│   ├── latent_noise.py
│   ├── post_process_color.py
│   ├── preprocess_sharpen_noise.py
│   └── post_process_sharpen_noise.py
└── your-collection/              # Your custom processors
    ├── processors.yaml (optional)
    └── *.py
```

## Creating a Processor Collection

### Option 1: With Manifest (Recommended)

Create a `processors.yaml` file listing your processors:

```yaml
processors:
  - my_processor.py
  - another_processor.py
```

### Option 2: Auto-Discovery

If no `processors.yaml` exists, all `.py` files in the folder will be auto-discovered (except files starting with `_` or named `base.py`/`setup.py`).

## Processor Requirements

1. **Use absolute imports:**
   ```python
   from streamdiffusion.preprocessing.processors import BasePreprocessor
   # NOT: from .base import BasePreprocessor
   ```

2. **Implement `get_preprocessor_metadata()` classmethod:**
   ```python
   @classmethod
   def get_preprocessor_metadata(cls):
       return {
           "display_name": "My Processor",
           "description": "What it does",
           "parameters": {
               "strength": {
                   "type": "float",
                   "default": 0.5,
                   "range": [0.0, 1.0],
                   "step": 0.01,
                   "description": "Effect strength"
               }
           },
           "use_cases": ["Use case 1", "Use case 2"]
       }
   ```

3. **Inherit from BasePreprocessor or PipelineAwareProcessor:**
   ```python
   class MyProcessor(BasePreprocessor):
       def __init__(self, strength: float = 0.5, **kwargs):
           super().__init__(strength=strength, **kwargs)
           self.strength = strength

       def _process_tensor_core(self, tensor):
           # Your processing logic
           return tensor * self.strength
   ```

## How Discovery Works

1. On module import, `src/streamdiffusion/preprocessing/processors/__init__.py` scans this folder
2. For each subfolder:
   - If `processors.yaml` exists, loads processors listed in the manifest
   - Otherwise, auto-discovers all `.py` files
3. Processors are dynamically imported and registered in `_preprocessor_registry`
4. TouchDesigner's `Getpreprocessors()` method uses subprocess to import the runtime registry
5. Metadata is extracted and Fx parameters are auto-generated

## Disabling Custom Processors

Set environment variable to disable discovery (useful for Docker/production):

```bash
export STREAMDIFFUSION_DISABLE_CUSTOM_PROCESSORS=1
```

## Naming Conventions

- **Folder name:** Use lowercase with underscores or hyphens (e.g., `my_processors` or `my-processors`)
- **Processor name:** Registry key derived from filename (e.g., `my_processor.py` → `my_processor`)
- **Class name:** PascalCase with `Preprocessor` suffix (e.g., `MyProcessorPreprocessor`)

**Avoid name conflicts with core processors!** Custom processors with names matching core processors will be rejected.

## Debugging

Check logs for discovery messages:
```
INFO: Scanning custom_processors/ folder for custom processors...
INFO: Loading processor collection 'csp' (4 processors)
INFO: Registered custom processor: latent_noise (LatentNoisePreprocessor)
```

## Examples

See `custom_processors/csp/` for working examples of:
- Latent-space noise injection (`latent_noise.py`)
- Image postprocessing color correction (`post_process_color.py`)
- Combined sharpen + noise effects (`preprocess_sharpen_noise.py`, `post_process_sharpen_noise.py`)

## Resources

- **Full Development Guide:** See `spencer_csp_processors/CLAUDE.md` for comprehensive processor development guide
- **Architecture Docs:** `claude_reports/multistage_processing_system_architecture.md`
- **Base Classes:** `src/streamdiffusion/preprocessing/processors/base.py`

## Support

For issues with custom processor discovery, check:
1. Import errors in logs
2. Metadata format (must return dict with required keys)
3. File naming (avoid `_` prefix, `base.py`, `setup.py`)
4. Absolute imports (not relative imports)
