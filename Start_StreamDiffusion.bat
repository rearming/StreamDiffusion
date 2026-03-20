
            @echo off
            cd /d %~dp0
            set HF_HOME=X:/td/StreamDiffusion/models
            if exist venv (
                call venv\Scripts\activate.bat
                venv\Scripts\python.exe streamdiffusionTD\td_main.py
            ) else (
                call .venv\Scripts\activate.bat
                .venv\Scripts\python.exe streamdiffusionTD\td_main.py
            )
            
            