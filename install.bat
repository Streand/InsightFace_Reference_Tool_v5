@echo off
REM Create venv if it doesn't exist
if not exist ".venv" (
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip to latest version
python -m pip install --upgrade pip

REM Install Python dependencies (with CUDA 11.8 wheels for torch/torchvision)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

REM Download and zip buffalo_l model
python download_buffalo_l.py

echo.
echo Setup complete! To activate your venv in the future, run:
echo     .venv\Scripts\activate
pause