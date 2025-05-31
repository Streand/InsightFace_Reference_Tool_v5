@echo off
REM Set CUDA 11.8 as the first in PATH for this session
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%"

REM Activate venv
call .venv\Scripts\activate

REM Display installed ONNX Runtime package and hardware type
echo ---------------------------------
for /f "delims=" %%i in ('python -c "import pkg_resources; pkgs=[p.project_name.lower() for p in pkg_resources.working_set]; print('NVIDIA GPUs (CUDA)' if 'onnxruntime-gpu' in pkgs else 'AMD/Intel GPUs (DirectML on Windows)' if 'onnxruntime-directml' in pkgs else 'CPU only' if 'onnxruntime' in pkgs else 'None')"') do set HW=%%i
echo Detected ONNX Runtime for: %HW%
python -c "import pkg_resources; print([p.project_name + '==' + p.version for p in pkg_resources.working_set if p.project_name.lower().startswith('onnxruntime')])"
echo ---------------------------------

REM Run the app and show error logs if any
echo Running InsightFace Reference Tool v5.3.2
echo ---------------------------------
python src/main.py
if errorlevel 1 (
    echo.
    echo [ERROR] The app failed to start. Check the error log above.
    pause
)

echo.
echo App is running. Check the URL in the output above.
echo Press Ctrl+C to stop the server.
pause