@echo off
REM Check if venv exists
if not exist ".venv\Scripts\activate" (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Uninstall all ONNX Runtime variants
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml

REM Ask user for hardware type
echo Select your hardware type for ONNX Runtime:
echo   1. NVIDIA GPU (CUDA)
echo   2. AMD/Intel GPU (DirectML)
echo   3. CPU only
set /p choice="Enter 1, 2, or 3: "

REM Install the correct ONNX Runtime package
if "%choice%"=="1" (
    pip install onnxruntime-gpu==1.15.1
) else if "%choice%"=="2" (
    pip install onnxruntime-directml
) else (
    pip install onnxruntime
)

echo.
echo ONNX Runtime package switched successfully!
pause