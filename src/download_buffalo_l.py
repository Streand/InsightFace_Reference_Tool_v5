import os
import sys
import time
import insightface

print("Downloading buffalo_l model to default InsightFace location...")

# Try multiple import strategies
try:
    # Try first approach with insightface.app
    from insightface.app import FaceAnalysis
    print("Successfully imported from insightface.app")
    analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
except ImportError:
    try:
        # Try direct model_zoo approach
        from insightface.model_zoo import get_model
        print("Using model_zoo.get_model() approach")
        model = get_model("buffalo_l")
        print("Model downloaded via model_zoo")
    except Exception as e:
        print(f"ERROR: All import attempts failed: {e}")
        print("Installed packages:")
        os.system("pip list")
        sys.exit(1)

# Only run this if using FaceAnalysis approach
if 'analyzer' in locals():
    try:
        print("Initializing analyzer to download model...")
        analyzer.prepare(ctx_id=-1)  # Use CPU for download
        print("Download initialization complete")
    except Exception as e:
        print(f"ERROR: Failed to prepare analyzer: {e}")
        sys.exit(1)

# Verify the model location
cache_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
if os.path.exists(cache_dir):
    print(f"Model successfully downloaded to: {cache_dir}")
    print("The application will use this model directly.")
else:
    print(f"WARNING: Could not find downloaded model at {cache_dir}")
    alt_paths = ['~/.insightface/buffalo_l', '~/.insightface/models']
    for path in alt_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"Model may be available at: {expanded_path}")

print("Download complete. Model will be loaded from the default location during runtime.")