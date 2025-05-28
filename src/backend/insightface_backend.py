import insightface
import os
import logging
import requests
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightFaceBackend:
    def __init__(self, model="buffalo_l", high_res=False, device="cuda:0"):
        """Initialize InsightFace backend with specified model and device

        Args:
            model (str): Model name - 'buffalo_l' (default)
            high_res (bool): Whether to use high resolution settings
            device (str): Device string, e.g. 'cuda:0' or 'cpu'
        """
        from insightface.app import FaceAnalysis

        self.model_name = model

        # Configure resolution
        det_size = (768, 768) if high_res else (640, 640)

        # Only support buffalo_l (remove antelopev2 and others)
        if model != "buffalo_l":
            raise ValueError("Only 'buffalo_l' model is supported in this version.")

        # Choose providers based on device string
        if device.startswith("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        elif device.startswith("directml"):
            providers = ['DirectMLExecutionProvider', 'CPUExecutionProvider']
            ctx_id = -1  # DirectML does not use CUDA context
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1

        # Try to load model
        try:
            logger.info(f"Loading InsightFace model: {model} on device: {device}")
            self.app = FaceAnalysis(name=model, providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            self.model = self.app
            logger.info(f"Successfully loaded model: {model}")
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error loading model {model}: {error_str}")
            
            # Check if this is a file not found error, which could happen after custom download
            if "No such file or directory" in error_str or "Cannot open file" in error_str:
                raise ValueError(f"Model files for {model} were downloaded but couldn't be loaded. The model structure may be incompatible with the current InsightFace version.")
            else:
                raise ValueError(f"Failed to load model {model}. Error: {error_str}")

    def get_embedding(self, img_path):
        """Get face embedding from an image path"""
        import cv2
        if not isinstance(img_path, (str, os.PathLike)):
            raise ValueError(f"Invalid image path: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        faces = self.model.get(img)
        if faces:
            return faces[0].embedding
        return None

    def compare(self, emb1, emb2):
        """Cosine similarity between two embeddings"""
        from numpy import dot
        from numpy.linalg import norm
        return dot(emb1, emb2) / (norm(emb1) * norm(emb2))