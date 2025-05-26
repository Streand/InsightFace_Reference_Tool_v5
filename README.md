# InsightFace Reference Tool v5

**InsightFace Reference Tool v5** is a user-friendly web app for finding the best-matching face images from a collection, powered by [InsightFace](https://github.com/deepinsight/insightface) facial recognition technology.

---

## Features

- **Reference & Input Images:** Upload one or more reference images and a folder of candidate images.
- **Face Matching:** Automatically finds and ranks the most similar faces in your dataset.
- **Customizable Settings:**
  - Minimum similarity threshold
  - Number of results to return
  - Option to use average embedding for references
  - High-resolution detection mode
  - Select processing device from the UI (GPU/CPU)
- **Progress Bar:** Real-time progress display during processing.
- **Download Results:** Download the best-matching images as a ZIP file.
- **Easy to Use:** Clean Gradio-based UI, no coding required.

---

## Requirements & Installation

- **Python 3.8–3.11** (recommended: 3.10+) and Microsoft C++ Build Tools [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/) During installation. choose: "Desktop development with C++" workload, which includes the MSVC v142/v143 compiler.
- You do **not** need to install any other packages, CUDA toolkit, or set environment variables globally.
- **All dependencies are installed automatically in a Python virtual environment (`.venv`) when you run `install.bat`.**
- **No system-wide Python changes are made.**
- Supported on Windows with Python 3.8–3.11 (recommended: 3.10+).
- CUDA 11.8 is required for GPU acceleration (the installer will fetch the correct PyTorch/torchvision wheels).

---

## Usage

1. **Upload reference images** (the person you want to find).
2. **Upload input images** (the folder to search).
3. **Adjust settings** as needed.
4. Click **Submit** to process and view/download results.

---

## Quick Start

1. **Clone the repository:**
    ```
    git clone https://github.com/Streand/InsightFace_Reference_Tool_v5.git
    ```

2. **Run the installer:**
    ```
    install.bat
    ```
    - This will create a `.venv` folder, install all requirements, and download the model if needed.

3. **Start the app:**
    ```
    run.bat
    ```
    - This will automatically activate the virtual environment and launch the app.

The application should open automatically in your default browser.

> **Note:**  
> The AI model for InsightFace will download/load after reference and input images have been added and submitted.  
> Model files are stored by default at:  
> `C:\Users\youruser\.insightface\models`

---

## Functions

- **Clear:** Clears all input and output fields.
- **Restart Script:** Restarts the current session and opens a new browser window.
- **Submit:** Runs the face matching process.

---

## Settings Explained

- **Similarity:**  
  Lower values will return more results (less strict matching).  
  Higher values will return only the best matches, but possibly fewer than requested.

- **How many Results (Can return less, if not enough matches):**  
  Set the number of results you want.  
  If there aren't enough matches above the similarity threshold, fewer images will be returned.

- **Use Average Embedding:**  
  Enable if you have multiple reference images.  
  Note: This may increase processing time and VRAM usage.

- **High-Resolution Mode:**  
  Uses higher resolution for detection (768px).  
  May improve results for large images, but will use more VRAM and take longer.

---

## Troubleshooting

- If the app does not open, check that your Python environment is set up correctly and that no other process is using the default port.
- The first run may take longer as the model is downloaded (see the log in `run.bat`).
- Please report issues, I'll do my best to fix them as soon as possible.


## known issues
- **Warning:**  
  If you see a message like  
  `NVIDIA GeForce RTX 50** with CUDA capability sm_120 is not compatible with the current PyTorch installation.`  
  you can safely ignore it. This is a false positive from PyTorch; the app should still work (even with RTX 5080). The warning is hidden by default, but may appear in some environments.
- if you do not have a nvidia gpu (cuda) you will get an error. But the software can be used with CPU. Currently only testd with Ryzen 99500x.
  I'm working on fixing the error.

---

## New Features & Fixes in v5

- **AMD GPU Support:**  
  The app now supports running on AMD GPUs (via ONNX Runtime, if available).
- **AMD CPU Support:**  
  Fully compatible with systems using only AMD CPUs.
- **Intel CPU Support:**  
  Fully compatible with systems using only Intel CPUs.
- **Improved Device Selection:**  
  You can now select the processing device (NVIDIA GPU, AMD GPU, Intel CPU, or AMD CPU) directly from the UI.
- **Cleaner Progress Bar:**  
  Real-time progress bar in the UI and improved terminal output.
- **Better Error Handling:**  
  Improved error messages and handling for missing models, incompatible files, and device issues.
- **General Stability Improvements:**  
  Various bug fixes and optimizations for smoother operation.

---

## Planned Features (v6 and Beyond)

- **Video Support:**  
  Ability to process video files and extract best-matching faces from video frames.  
  *(More information will be added as development progresses.)*

---

## License

This project is for research and personal use. See [InsightFace license](https://github.com/deepinsight/insightface/blob/master/LICENSE) for model terms.
