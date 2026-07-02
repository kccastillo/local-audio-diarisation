@echo off
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag an audio file onto this launcher to transcribe it.
    pause
    exit /b 0
)

if not "%~2"=="" (
    echo Note: only the first dropped file is processed.
)

"%~dp0venv\Scripts\python.exe" -m diarizer.cli run --input "%~1"

pause
