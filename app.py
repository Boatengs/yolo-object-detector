
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
from ultralytics import YOLO

# Load model once at startup
model = YOLO("yolov8l.pt")

def detect_objects(image, confidence, model_size):
    if image is None:
        return None, "Please upload an image first!"

    yolo = YOLO(f"yolov8{model_size}.pt")
    img_np = np.array(image)
    results = yolo(img_np, conf=confidence / 100, verbose=False)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    detections = {}
    for box in boxes:
        cls_name = yolo.names[int(box.cls)]
        detections[cls_name] = detections.get(cls_name, 0) + 1

    total = sum(detections.values())
    if total == 0:
        summary = "No objects detected. Try lowering the confidence threshold."
    else:
        summary = f" {total} object(s) detected:\n\n"
        for obj, count in sorted(detections.items(), key=lambda x: -x[1]):
            summary += f"   {obj}: {count}\n"

    return Image.fromarray(annotated_rgb), summary

def load_street():
    img = Image.open(BytesIO(
        requests.get("https://ultralytics.com/images/bus.jpg").content
    )).convert("RGB")
    return detect_objects(img, 25, "l")

def load_sports():
    img = Image.open(BytesIO(
        requests.get("https://ultralytics.com/images/zidane.jpg").content
    )).convert("RGB")
    return detect_objects(img, 25, "l")

with gr.Blocks(title="YOLOv8 Object Detector") as demo:

    gr.Markdown("""
    # ⚡ YOLOv8 Object Detector
    Upload any image and detect objects instantly using YOLOv8 trained on COCO (80 classes).
    Works on street scenes, sports, wildlife, kitchen, people and more!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Any Image", height=320)
            confidence  = gr.Slider(minimum=10, maximum=90, value=25, step=5, label="Confidence Threshold (%)")
            model_size  = gr.Radio(choices=["n", "s", "m", "l"], value="l", label="Model Size (n=fastest, l=most accurate)")
            detect_btn  = gr.Button("Detect Objects", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Detection Result", height=320)
            output_text  = gr.Textbox(label="Detection Summary", lines=10)

    gr.Markdown("### Try these examples:")
    with gr.Row():
        street_btn = gr.Button(" Street Scene")
        sports_btn = gr.Button(" Sports Scene")

    detect_btn.click(fn=detect_objects, inputs=[input_image, confidence, model_size], outputs=[output_image, output_text])
    street_btn.click(fn=load_street, inputs=None, outputs=[output_image, output_text])
    sports_btn.click(fn=load_sports, inputs=None, outputs=[output_image, output_text])

demo.launch()
