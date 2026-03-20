@echo off
cd /d X:\td\StreamDiffusion
call venv\Scripts\activate.bat
python test_osc_listen.py
pause
