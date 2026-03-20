@echo off
cd /d X:\td\StreamDiffusion
call venv\Scripts\activate.bat
python test_spout_api.py > spout_api.txt 2>&1
type spout_api.txt
pause
