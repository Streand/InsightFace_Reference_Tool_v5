# InsightFace Reference Tool v5.3.2

**InsightFace Reference Tool v5.3.2** is a user-friendly web app for finding the best-matching face images from a collection, powered by [InsightFace](https://github.com/deepinsight/insightface) facial recognition technology.

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
- **Support:** For NVIDIA, AMD GPUs and CPUs.
---
## UI
![Image](https://github.com/user-attachments/assets/c9c7a11a-00e3-4c99-a8d9-a21e051b9d7b)

## Results

In this example, I tested the app with a dataset of **14,000 high-quality, real photos**.  
I added **5 images of Anna Kendrick** as wildcards to the input images, each with different resolutions and aspect ratios (including 1 duplicate to test detection).  
For the reference image, I used a different photo of Anna Kendrick.

- **Settings:** Default similarity, requested 10 results (but only 5 wildcards were present).
- **Result:** The tool successfully found all Anna Kendrick images, even with only 5 wildcards in the set.
- **Performance:** The search completed in just **363 seconds** on my setup.

This demonstrates the tool’s accuracy and efficiency, even with large, diverse datasets.

![Image](https://github.com/user-attachments/assets/90437a1d-cb90-4b3a-a844-2cd5b398458e)

![Image](https://github.com/user-attachments/assets/33bbc7cf-d213-4549-871a-2aada0cff6e8)

---

## Requirements & Installation

- **Python 3.10.** and Microsoft C++ Build Tools [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/) During installation. choose: "Desktop development with C++" workload, which includes the MSVC v142/v143 compiler.
- You do **not** need to install any other packages, CUDA toolkit, or set environment variables globally.
- **All dependencies are installed automatically in a Python virtual environment (`.venv`) when you run `install.bat`.**
- **No system-wide Python changes are made.**
- Supported on Windows with Python 3.10.*

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
    - Choose the GPU you are using.
    - NOTE: use 'switch_gpu.bat' to if you have installed wrong or harware change. it will uninstall the current version and install correct depeding on optiop you choose.
  

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
- **Clean Gradio Temp:** will clear temp .gradio folder, This will also clean other applications that uses Gradio 
- **Processing Device:** Dropdown menu for gpu / cpu you want to run the proccess with


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

---

## Planned Features & Fixes for v5.x.x ->

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

---

## change log and fixes

- **5.0.1**
  - added clear gradio temp function, made code run faster on startup
- **5.0.2**
  - CUDA capabilities error / bug
- **5.1.0**
  - fix no cuda option error
- **5.2.0**
  - adding AMD support and other major code changes.
  - add warning to clear gradio cache.
  - adding GPU switcher
  - better install.bat functions
  - new info to readme
- **5.3.0**
  - multiple errors and follow up issues, after 5.2.0 should have been fixed
  - updated to InsightFace 0.7.3 -> 0.9.1
  - adding more decencies to make muli-option device work correctly
  - corrected pathing issue for model downloader
  - adding better logs during installation
  - isolating requirement specific for gpu option
  - added CPU name detector
- **5.3.1**
  -  fixed: port already in use
- **5.3.2**
  - added better error in UI
  - added Face Analysis Visualization in UI