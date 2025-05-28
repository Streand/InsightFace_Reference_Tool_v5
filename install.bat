@echo off
REM Create venv if it doesn't exist
if not exist ".venv" (
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip to latest version
python -m pip install --upgrade pip

REM Ask user for hardware type
echo Select your hardware type for ONNX Runtime:
echo   1. NVIDIA GPU (CUDA)
echo   2. AMD/Intel GPU (DirectML)
echo   3. CPU only
set /p choice="Enter 1, 2, or 3: "

REM Install base requirements (excluding ONNX Runtime)
pip install gradio insightface==0.7.3 numpy==1.26.4 opencv-python Pillow==9.0.0 torch==2.0.1+cu118 torchvision==0.15.2+cu118 scikit-image==0.19.3 matplotlib==3.5.1 scikit-learn fastapi

REM Install the correct ONNX Runtime package
if "%choice%"=="1" (
    pip install onnxruntime-gpu==1.15.1
) else if "%choice%"=="2" (
    pip install onnxruntime-directml
) else (
    pip install onnxruntime
)

REM Download and zip buffalo_l model
python download_buffalo_l.py

echo.
echo Setup complete! To activate your venv in the future, run:
echo     .venv\Scripts\activate
pause