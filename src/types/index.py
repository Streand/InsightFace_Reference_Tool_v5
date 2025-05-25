from typing import List, Dict, Any

class ImageData:
    def __init__(self, file_path: str, quality_score: float):
        self.file_path = file_path
        self.quality_score = quality_score

class ProcessingResult:
    def __init__(self, processed_images: List[ImageData], summary: Dict[str, Any]):
        self.processed_images = processed_images
        self.summary = summary