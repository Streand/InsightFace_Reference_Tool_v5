import insightface
import shutil
import os
import zipfile

# Download model to default cache
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

# Find the downloaded model directory
cache_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
target_dir = r'G:\AI2\loradatahelper\ai-recognition-app\models\buffalo_l'
zip_path = r'G:\AI2\loradatahelper\ai-recognition-app\models\buffalo_l.zip'

# Copy model files to your models directory
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.copytree(cache_dir, target_dir)

# Zip the model folder
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(target_dir))
            zipf.write(file_path, arcname)

print(f"buffalo_l model downloaded and zipped at: {zip_path}")