from gradio import Interface, inputs, outputs
import os
from utils.image_processing import process_images
from backend.insightface_backend import InsightFaceBackend

def create_ui():
    backend = InsightFaceBackend()

    def process_and_display(images):
        results = process_images(images, backend)
        return results

    image_input = inputs.Image(type="filepath", label="Upload Images", tool="editor", multiple=True)
    output_display = outputs.Image(type="numpy", label="Processed Images")

    interface = Interface(
        fn=process_and_display,
        inputs=image_input,
        outputs=output_display,
        title="AI Recognition App",
        description="Upload images to process and find the best ones for LoRa training.",
        theme="default"
    )

    return interface

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)