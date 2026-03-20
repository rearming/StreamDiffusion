@echo off
cd /d X:\td\StreamDiffusion
chcp 65001 >nul

set HF_HOME=X:/hf_cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8

call venv\Scripts\activate.bat

echo === Using manual config (bypassing .tox) ===
copy /Y StreamDiffusionTD\td_config_manual.yaml StreamDiffusionTD\td_config.yaml

venv\Scripts\python.exe StreamDiffusionTD\td_main.py

pause
