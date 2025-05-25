import os
import zipfile
import tempfile
import shutil
import sys
from PIL import Image
import numpy as np
import gradio as gr

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def resize_image(image, target_size=(256, 256)):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)

def evaluate_image_quality(image):
    # Placeholder for image quality evaluation logic
    # This could involve checking for sharpness, brightness, etc.
    return True  # Assume all images are of good quality for now

def process_images(image_paths):
    processed_images = []
    for path in image_paths:
        image = load_image(path)
        resized_image = resize_image(image)
        if evaluate_image_quality(resized_image):
            processed_images.append(resized_image)
    return processed_images

def process_images_and_zip(og_images, folder_images, backend, top_k=8, min_similarity=0.3, use_avg_embedding=False, progress=gr.Progress(track_tqdm=True)):
    # Ensure og_images is a list (to support multiple OG images)
    if not isinstance(og_images, list):
        og_images = [og_images]

    # Get embeddings for all OG images
    og_embs = []
    for og_image in og_images:
        if og_image:  # Ensure the file path is valid
            emb = backend.get_embedding(og_image)
            if emb is not None:
                og_embs.append(emb)
    if not og_embs:
        raise ValueError("No valid embeddings found for reference images.")
    
    # Create average embedding if option is selected
    if use_avg_embedding and len(og_embs) > 1:
        avg_emb = np.mean(og_embs, axis=0)
        # Normalize the embedding
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        og_embs = [avg_emb]  # Replace with just the average embedding

    # Process candidate images
    scored = []
    total = len(folder_images)
    for idx, img_path in enumerate(folder_images, 1):
        try:
            emb = backend.get_embedding(img_path)
            if emb is not None:
                # Compare to all OG embeddings, take max similarity
                similarities = [backend.compare(emb, ref_emb) for ref_emb in og_embs]
                best_sim = max(similarities)
                if best_sim >= min_similarity:
                    scored.append((best_sim, img_path))
        except Exception as e:
            print(f"\nError processing {img_path}: {e}", file=sys.stderr)
        # Terminal progress bar
        bar = "[" + "#" * int(40 * idx / total) + "-" * (40 - int(40 * idx / total)) + "]"
        print(f"\rProcessing images {bar} {idx}/{total}", end="", flush=True)
        # Gradio progress
        if progress is not None:
            progress(idx / total)
    print()  # Newline after progress bar

    # Sort by similarity and select top_k
    scored.sort(reverse=True)
    best_images = [img for _, img in scored[:top_k]]

    # Create a zip file with best images
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "best_images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img in best_images:
            zipf.write(img, os.path.basename(img))
    return best_images, zip_path