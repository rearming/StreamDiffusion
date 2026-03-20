@echo off
cd /d X:\td\StreamDiffusion
chcp 65001 >nul

set HF_HOME=X:/hf_cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8

call venv\Scripts\activate.bat
python server_minimal.py
pause
