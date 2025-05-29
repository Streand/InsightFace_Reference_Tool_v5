@echo off
echo activatiing virtual environment...
REM Create venv if it doesn't exist
if not exist ".venv" (
    python -m venv .venv
)


REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip to latest version
python -m pip install --upgrade pip

REM Explicitly install setuptools and wheel first
python -m pip install --upgrade setuptools<81.0 wheel

REM Ensure InsightFace is properly installed with dependencies
pip uninstall -y insightface
pip install --upgrade insightface onnx numpy==1.26.4 opencv-python

REM Ask user for hardware type
echo Select your hardware type for ONNX Runtime:
echo   1. NVIDIA GPU (CUDA)
echo   2. AMD/Intel GPU (DirectML)
echo   3. CPU only
set /p choice="Enter 1, 2, or 3: "

REM Choose PyTorch installation based on GPU selection
if "%choice%"=="1" (
    echo.
    echo NOTE: Installing NVIDIA CUDA packages will take several minutes...
    echo The system may appear to freeze at "Looking in indexes" - this is normal.
    echo Please be patient and don't close this window.
    echo.
    REM Install NVIDIA-compatible PyTorch with CUDA 11.8
    pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    pip install onnxruntime-gpu==1.15.1
) else if "%choice%"=="2" (
    echo.
    echo Installing DirectML packages for AMD/Intel GPUs...
    echo.
    REM Install CPU version of PyTorch for DirectML
    pip install torch==2.2.2 torchvision==0.17.2
    pip install onnxruntime-directml
) else (
    echo.
    echo Installing CPU-only packages...
    echo.
    REM Install CPU version of PyTorch
    pip install torch==2.2.2 torchvision==0.17.2
    pip install onnxruntime
)

REM Install Gradio packages with their dependencies first
echo Installing Gradio and UI dependencies...
pip install gradio_client gradio --upgrade

REM Install other requirements (without torch/torchvision)
pip install -r requirements.txt --no-deps

REM Download and zip buffalo_l model
if exist src\download_buffalo_l.py (
    python src\download_buffalo_l.py
) else if exist download_buffalo_l.py (
    python download_buffalo_l.py
)

echo.
echo Setup complete! To activate your venv in the future, run:
echo     .venv\Scripts\activate
pause