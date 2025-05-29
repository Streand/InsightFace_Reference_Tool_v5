import insightface
from insightface.app import FaceAnalysis
import os
import logging
import requests
import shutil
import onnxruntime 
import numpy as np
from numpy.linalg import norm
import cv2 
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
            device (str): Device string, e.g. 'cuda:0', 'directml', or 'cpu'
        """

        self.model_name = model
        self.device = device  # Store for debugging

        # Configure resolution
        det_size = (768, 768) if high_res else (640, 640)

        # Only support buffalo_l (remove antelopev2 and others)
        if model != "buffalo_l":
            raise ValueError("Only 'buffalo_l' model is supported in this version.")

        # Print available providers for debugging
        logger.info(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")
        logger.info(f"Requested device: {device}")

        # Choose providers based on device string - FIXED PROVIDER NAMES
        if device.lower().startswith("cuda"):
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            logger.info("Using NVIDIA GPU with CUDA")
        elif device.lower().startswith("directml"):
            self.providers = ['DmlExecutionProvider', 'CPUExecutionProvider']  # FIXED name
            ctx_id = 0  # Try using 0 for DirectML
            logger.info("Using AMD/Intel GPU with DirectML")
        else:
            self.providers = ['CPUExecutionProvider']
            ctx_id = -1
            logger.info("Using CPU only")

        # Try to load model
        try:
            logger.info(f"Loading InsightFace model: {model} with providers: {self.providers}")
            self.app = FaceAnalysis(name=model, providers=self.providers)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            self.model = self.app
            
            # Verify which provider is actually being used
            logger.info(f"Model loaded successfully. Provider chain: {self.providers}")
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
        try:
            if not isinstance(img_path, (str, os.PathLike)):
                raise ValueError(f"Invalid image path: {img_path}")
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            faces = self.model.get(img)
            if faces:
                return faces[0].embedding
            logger.warning(f"No faces detected in image: {img_path}")
            return None
        except Exception as e:
            logger.error(f"Error extracting embedding from {img_path}: {str(e)}")
            return None

    def compare(self, emb1, emb2):
        """Cosine similarity between two embeddings"""
        try:
            # Using imports from the top of file now
            # No need for: from numpy import dot, from numpy.linalg import norm
            
            # Validate embeddings
            if emb1 is None or emb2 is None:
                return 0.0
                
            n1 = norm(emb1)
            n2 = norm(emb2)
            
            # Avoid division by zero
            if n1 == 0 or n2 == 0:
                return 0.0
                
            return np.dot(emb1, emb2) / (n1 * n2)  # Changed to np.dot
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return 0.0