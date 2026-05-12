
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import requests
from io import BytesIO

#  Load Model 
model = YOLO("yolov8l.pt")

def detect_objects(image, confidence, model_size):
    if image is None:
        return None, "Please upload an image first!"

    yolo   = YOLO(f"yolov8{model_size}.pt")
    img_np = np.array(image)

    results    = yolo(img_np, conf=confidence / 100, verbose=False)
    annotated  = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes      = results[0].boxes
    detections = {}
    for box in boxes:
        cls_name = yolo.names[int(box.cls)]
        detections[cls_name] = detections.get(cls_name, 0) + 1

    total = sum(detections.values())
    if total == 0:
        summary = " No objects detected. Try lowering the confidence threshold."
    else:
        summary = f"###  {total} Object(s) Detected\n\n"
        for obj, count in sorted(detections.items(), key=lambda x: -x[1]):
            bar = "█" * min(count * 3, 20)
            summary += f"**{obj}** — {count}x {bar}\n\n"

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

# ── CSS ────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #ffffff;
    --bg-elevated: #f1f5f9;
    --border: rgba(0,0,0,0.08);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-tertiary: #94a3b8;
    --accent: #f59e0b;
    --accent-hover: #d97706;
    --accent-dim: rgba(245,158,11,0.1);
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08);
    --radius: 16px;
}

[data-theme="dark"] {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --bg-elevated: #334155;
    --border: rgba(255,255,255,0.08);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-tertiary: #475569;
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4);
}

* { font-family: "Inter", sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease;
}

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 24px !important; }

.app-header {
    text-align: center;
    padding: 48px 24px 32px;
    background: linear-gradient(135deg, var(--accent-dim), transparent);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header h1 {
    font-size: 2.4em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent), #ef4444);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 12px !important;
}
.app-header p {
    color: var(--text-secondary) !important;
    font-size: 1em !important;
    max-width: 600px;
    margin: 0 auto !important;
    line-height: 1.7 !important;
}
.badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.badge {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--bg-elevated);
    color: var(--text-secondary);
}
.badge.accent { background: var(--accent-dim); color: var(--accent); border-color: rgba(245,158,11,0.25); }

.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
}

.theme-toggle {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    cursor: pointer !important;
    font-size: 1.2em !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.2s !important;
    z-index: 1000 !important;
}
.theme-toggle:hover { transform: scale(1.1) !important; }

.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    box-shadow: var(--shadow);
}
.stat-value { font-size: 1.8em; font-weight: 800; color: var(--accent); }
.stat-label { font-size: 0.75em; color: var(--text-tertiary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

button.primary {
    background: linear-gradient(135deg, var(--accent), #ef4444) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(245,158,11,0.35) !important;
}
button.primary:hover { transform: translateY(-1px) !important; }

label { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.82em !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }

.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-tertiary);
    font-size: 0.78em;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}
"""

js = """
function toggleTheme() {
    const root = document.documentElement;
    const btn  = document.getElementById("theme-btn");
    const dark = root.getAttribute("data-theme") === "dark";
    root.setAttribute("data-theme", dark ? "light" : "dark");
    btn.textContent = dark ? "🌙" : "☀️";
    localStorage.setItem("yolo-theme", dark ? "light" : "dark");
}
document.addEventListener("DOMContentLoaded", () => {
    const saved = localStorage.getItem("yolo-theme") || "light";
    document.documentElement.setAttribute("data-theme", saved);
    const btn = document.getElementById("theme-btn");
    if (btn) btn.textContent = saved === "dark" ? "☀️" : "🌙";
});
"""

with gr.Blocks(css=css, title="⚡ YOLOv8 Object Detector") as demo:

    gr.HTML(f"""
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">🌙</button>
    <script>{js}</script>

    <div class="app-header">
        <h1>⚡ YOLOv8 Object Detector</h1>
        <p>
            Real-time object detection powered by YOLOv8 Large.
            Upload any image and instantly detect objects across 80 COCO classes
            — people, vehicles, animals, food, furniture and more.
        </p>
        <div class="badges">
            <span class="badge accent">YOLOv8 Large</span>
            <span class="badge accent">80 Classes</span>
            <span class="badge">43.7M Parameters</span>
            <span class="badge">Real-time Detection</span>
            <span class="badge">COCO Dataset</span>
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">80</div>
            <div class="stat-label">Object Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">43.7M</div>
            <div class="stat-label">Parameters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Model Sizes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">95%</div>
            <div class="stat-label">Top Confidence</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("###  Upload Image")
            input_image = gr.Image(
                type="pil",
                label="Upload Any Image",
                height=280
            )
            confidence = gr.Slider(
                minimum=10, maximum=90,
                value=25, step=5,
                label="Confidence Threshold (%)"
            )
            model_size = gr.Radio(
                choices=["n", "s", "m", "l"],
                value="l",
                label="Model Size (n=fastest · l=most accurate)"
            )
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
            gr.Markdown("""
            <div style="margin-top:12px; padding:12px; background:var(--bg-elevated); border-radius:10px; font-size:0.8em; color:var(--text-secondary);">
            💡 <b>Try these quick examples:</b>
            </div>
            """)
            with gr.Row():
                street_btn = gr.Button(" Street Scene", size="sm")
                sports_btn = gr.Button(" Sports Scene", size="sm")

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("### 🎯 Detection Results")
            output_image = gr.Image(label="Annotated Image", height=280)
            output_text  = gr.Markdown(
                value="*Upload an image and click Detect Objects to see results*"
            )

    gr.HTML("""
    <div style="margin-top:28px; background:var(--bg-card); border:1px solid var(--border); border-radius:16px; padding:24px; box-shadow:var(--shadow);">
        <h3 style="color:var(--text-primary); margin-bottom:16px; font-size:1em; font-weight:700;">📋 Detectable Object Categories</h3>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px;">
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">👤</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">People</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">person</div>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">🚗</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">Vehicles</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">car, bus, truck, bike</div>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">🐕</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">Animals</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">dog, cat, horse, bear</div>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">🍕</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">Food</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">pizza, apple, banana</div>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">💻</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">Electronics</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">laptop, phone, TV</div>
            </div>
            <div style="padding:12px; background:var(--bg-elevated); border-radius:10px; text-align:center;">
                <div style="font-size:1.5em;">🛋️</div>
                <div style="font-weight:600; color:var(--text-primary); font-size:0.85em; margin-top:4px;">Furniture</div>
                <div style="color:var(--text-tertiary); font-size:0.72em;">chair, couch, bed</div>
            </div>
        </div>
    </div>

    <div class="footer">
        Built with YOLOv8 Large • Trained on COCO Dataset • 80 Object Classes •
        <a href="https://github.com/Boatengs/yolo-object-detector" target="_blank" style="color:var(--accent);">GitHub</a>
    </div>
    """)

    detect_btn.click(fn=detect_objects, inputs=[input_image, confidence, model_size], outputs=[output_image, output_text])
    street_btn.click(fn=load_street, inputs=None, outputs=[output_image, output_text])
    sports_btn.click(fn=load_sports, inputs=None, outputs=[output_image, output_text])

demo.launch()
