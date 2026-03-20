
        @echo off
        echo.
        echo ========================================
        echo  StreamDiffusionTD v0.3.0 Installation
        echo ========================================
        echo.
        cd /d "X:/td/StreamDiffusion"
        echo Changed directory to: %CD%
        set "PIP_DISABLE_PIP_VERSION_CHECK=1"

        if exist "venv" (
            echo Clearing pip cache before update...
            python -m pip cache purge
        )

        if not exist "venv" (
            echo Creating Python venv at: "X:/td/StreamDiffusion\venv"
            "X:/td/StreamDiffusion\venv\Scripts\python.exe" -m venv venv
        ) else (
            echo Virtual environment already exists at: "X:/td/StreamDiffusion\venv"
        )

        echo Attempting to activate virtual environment...
        call "venv\Scripts\activate.bat"

        rem Check if the virtual environment was activated successfully
        if "%VIRTUAL_ENV%" == "" (
            echo Failed to activate virtual environment.
            pause
            exit /b 1
        ) else (
            echo Virtual environment activated.
        )

        echo.
        echo.
        echo [1/7] Base System Setup
        echo Installing pip, setuptools, wheel...
        python -m pip install  --upgrade pip setuptools wheel

        echo Installing compatible NumPy first (fixes NumPy 2.x conflicts)...
        python -m pip install  "numpy<2.0.0"

        echo.
        echo.
        echo [2/7] CUDA Stack Installation
        echo Installing PyTorch with CUDA support...
        python -m pip install  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && python -m pip install  cuda-python==12.9.0

        echo.
        echo Verifying CUDA PyTorch installation...
        python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')" || echo "WARNING: PyTorch verification failed"

        echo.
        echo.
        echo [3/7] Core Dependencies
        echo Installing diffusers, transformers, accelerate...
        python -m pip install  --no-deps diffusers transformers accelerate omegaconf protobuf

        echo Installing missing dependencies that don't conflict with torch...
        python -m pip install  safetensors huggingface_hub regex requests tqdm filelock packaging pyyaml

        echo.
        echo.
        echo [4/7] StreamDiffusion Installation
        echo Installing DotSimulate StreamDiffusion fork from local clone (editable mode)...
        python -m pip install  --no-deps -e .[tensorrt]

        echo Installing Diffusers IPAdapter (no deps)...
        python -m pip install  --no-deps git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99

        echo.
        echo.
        echo [5/7] Computer Vision Stack
        echo Installing opencv and image processing libraries...
        python -m pip install  opencv-python==4.8.1.78 Pillow scipy scikit-image

        echo Installing controlnet_aux WITHOUT dependencies (prevents torch conflicts)...
        python -m pip install  --no-deps controlnet_aux

        echo Installing controlnet_aux missing dependencies manually...
        python -m pip install  timm mediapipe

        echo.
        echo.
        echo [6/7] TouchDesigner Integration
        echo Installing TouchDesigner-specific packages...
        python -m pip install  python-osc pywin32 fire mss einops peft>=0.17.0

        echo Installing matplotlib (large dependency tree, but safe after torch is locked)...
        python -m pip install  matplotlib

        echo Installing insightface (optional, may have conflicts)...
        python -m pip install  insightface || echo "WARNING: insightface installation failed - this is optional for FaceID"

        echo.
        echo.
        echo [7/7] Final Verification
        echo Fixing version conflicts...
        python -m pip install  "numpy<2.0.0" --force-reinstall

        echo Installing optional performance packages...
        python -m pip install  triton || echo "INFO: triton not available - this is normal on Windows"

        echo.
        echo.
        echo ========================================
        echo Verification
        echo ========================================
        echo Checking PyTorch CUDA...
        python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA not available!'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || echo "CUDA verification FAILED"

        echo Checking StreamDiffusion...
        python -c "from streamdiffusion.config import load_config; print('StreamDiffusion installed successfully')" || echo "StreamDiffusion verification FAILED"

        echo.
        echo ========================================
        echo Installation Summary
        echo ========================================
        python -m pip list | findstr /I "torch diffusers streamdiffusion xformers cuda-python"

        echo.
        echo ========================================
        echo Installation Complete
        echo ========================================
        echo Check verification results above.
        echo.
        pause
        