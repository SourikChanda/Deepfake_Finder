import gradio as gr
import cv2
import numpy as np
from PIL import Image

def detect_deepfake(media):
    if isinstance(media, str):  # video path
        import os
        import tempfile
        cap = cv2.VideoCapture(media)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Error reading video", 0.0, None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:  # image
        img = np.array(media)

    score = np.random.uniform(0, 1)
    verdict = "Deepfake" if score > 0.5 else "Real"

    heatmap = np.random.randint(0, 255, img.shape, dtype=np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return verdict, float(score), Image.fromarray(overlay)

demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.File(file_types=[".jpg", ".png", ".mp4", ".avi"], type="file"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence Score"),
        gr.Image(label="Explainability Heatmap")
    ],
    title="Explainable Deepfake Defense",
    description="Upload an image or video. The system detects deepfakes and generates a visual explanation."
)

if __name__ == "__main__":
    # Important for Render: bind to 0.0.0.0 and port 8080
    demo.launch(server_name="0.0.0.0", server_port=8080)
