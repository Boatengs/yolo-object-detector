
import gradio as gr
import torch
import requests
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from io import BytesIO

# Load Grounding DINO model
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id  = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()

def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")

def detect_objects(image, labels_text, threshold):
    if image is None:
        return None, "Please upload an image."

    # Parse labels and build text prompt
    labels      = [l.strip() for l in labels_text.split(",") if l.strip()]
    text_prompt = " . ".join(labels) + " ."

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        target_sizes=[image.size[::-1]]
    )[0]

    results["labels"] = results["text_labels"]

    # Draw detections
    img_draw = image.copy()
    draw     = ImageDraw.Draw(img_draw)

    colors = ["#6366f1", "#10b981", "#ef4444", "#f59e0b",
              "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6",
              "#f97316", "#06b6d4", "#84cc16", "#a855f7"]

    boxes  = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    # Count detections per label
    label_counts = {}
    for label in labels:
        if label.strip():
            label_counts[label] = label_counts.get(label, 0) + 1

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if not label.strip():
            continue
        color   = colors[i % len(colors)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"
        draw.rectangle([x1, y1 - 20, x1 + len(text) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 18), text, fill="white")

    # Build summary
    total   = len([l for l in labels if l.strip()])
    summary = f"### Detected {total} object(s)\n\n"

    counts = {}
    for label in labels:
        if label.strip():
            counts[label] = counts.get(label, 0) + 1

    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar     = "█" * min(count * 4, 24)
        summary += f"**{label}** — {count}x {bar}\n\n"

    if total == 0:
        summary = "No objects detected. Try lowering the threshold or adding more specific labels."

    return img_draw, summary


# Preset label groups for quick access
presets = {
    "Street Scene"      : "person, car, bus, truck, bicycle, traffic light, traffic sign, building, road",
    "Kitchen"           : "microwave, oven, refrigerator, sink, stove, cabinet, counter, toaster, kettle",
    "Living Room"       : "sofa, chair, table, television, lamp, bookshelf, carpet, cushion, window",
    "Space & Galaxy"    : "galaxy, star, nebula, planet, spiral galaxy, moon, comet, asteroid",
    "Animals"           : "dog, cat, bird, horse, elephant, lion, tiger, bear, rabbit, fish",
    "Sports"            : "person, ball, goal, net, court, stadium, athlete, referee, jersey",
    "Office"            : "laptop, monitor, keyboard, mouse, desk, chair, phone, printer, notebook",
    "Nature"            : "tree, flower, grass, mountain, river, rock, cloud, sky, waterfall, forest",
}

css = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap");
:root {
    --bg-primary:#ffffff; --bg-card:#ffffff; --bg-elevated:#f1f5f9;
    --border:rgba(0,0,0,0.08); --text-primary:#0f172a; --text-secondary:#475569;
    --text-tertiary:#94a3b8; --accent:#f59e0b; --accent-dim:rgba(245,158,11,0.1);
    --shadow:0 4px 6px -1px rgba(0,0,0,0.07); --shadow-lg:0 10px 15px -3px rgba(0,0,0,0.08);
    --radius:16px;
}
[data-theme="dark"] {
    --bg-primary:#0f172a; --bg-card:#1e293b; --bg-elevated:#334155;
    --border:rgba(255,255,255,0.08); --text-primary:#f1f5f9;
    --text-secondary:#94a3b8; --text-tertiary:#475569;
    --shadow:0 4px 6px -1px rgba(0,0,0,0.3); --shadow-lg:0 10px 15px -3px rgba(0,0,0,0.4);
}
* { font-family:"Inter",sans-serif !important; box-sizing:border-box; }
body,.gradio-container { background:var(--bg-primary) !important; color:var(--text-primary) !important; transition:all 0.3s ease; }
.gradio-container { max-width:1100px !important; margin:0 auto !important; padding:24px !important; }
.app-header { text-align:center; padding:52px 24px 36px; background:linear-gradient(135deg,var(--accent-dim),transparent); border-radius:var(--radius); border:1px solid var(--border); margin-bottom:28px; }
.app-header h1 { font-size:2.4em !important; font-weight:900 !important; background:linear-gradient(135deg,var(--accent),#ef4444,#8b5cf6); -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; margin-bottom:14px !important; }
.app-header p { color:var(--text-secondary) !important; font-size:1em !important; max-width:640px; margin:0 auto !important; line-height:1.8 !important; }
.badges { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:20px; }
.badge { padding:6px 16px; border-radius:20px; font-size:0.75em; font-weight:600; border:1px solid var(--border); background:var(--bg-elevated); color:var(--text-secondary); }
.badge.accent { background:var(--accent-dim); color:var(--accent); border-color:rgba(245,158,11,0.25); }
.badge.positive { background:rgba(16,185,129,0.1); color:#10b981; border-color:rgba(16,185,129,0.25); }
.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
.stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px 16px; text-align:center; box-shadow:var(--shadow); transition:all 0.2s ease; }
.stat-card:hover { transform:translateY(-2px); box-shadow:var(--shadow-lg); }
.stat-value { font-size:1.9em; font-weight:900; color:var(--accent); line-height:1; }
.stat-label { font-size:0.7em; color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-top:6px; }
.card { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important; padding:24px !important; box-shadow:var(--shadow) !important; }
.theme-toggle { position:fixed !important; top:20px !important; right:20px !important; background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:50% !important; width:46px !important; height:46px !important; cursor:pointer !important; font-size:1.2em !important; box-shadow:var(--shadow-lg) !important; transition:all 0.2s !important; z-index:1000 !important; }
.theme-toggle:hover { transform:scale(1.12) rotate(15deg) !important; }
button.primary { background:linear-gradient(135deg,var(--accent),#ef4444) !important; border:none !important; border-radius:12px !important; color:white !important; font-weight:700 !important; font-size:1em !important; padding:14px 28px !important; width:100% !important; cursor:pointer !important; transition:all 0.25s !important; box-shadow:0 4px 15px rgba(245,158,11,0.4) !important; }
button.primary:hover { transform:translateY(-2px) !important; box-shadow:0 8px 25px rgba(245,158,11,0.5) !important; }
label { color:var(--text-secondary) !important; font-weight:700 !important; font-size:0.78em !important; text-transform:uppercase !important; letter-spacing:1px !important; }
.footer { text-align:center; padding:28px; color:var(--text-tertiary); font-size:0.78em; border-top:1px solid var(--border); margin-top:36px; line-height:2; }
"""

js = """
function toggleTheme(){
    const root=document.documentElement;
    const btn=document.getElementById("theme-btn");
    const dark=root.getAttribute("data-theme")==="dark";
    root.setAttribute("data-theme",dark?"light":"dark");
    btn.textContent=dark?"🌙":"☀️";
    localStorage.setItem("gdino-theme",dark?"light":"dark");
}
document.addEventListener("DOMContentLoaded",()=>{
    const saved=localStorage.getItem("gdino-theme")||"light";
    document.documentElement.setAttribute("data-theme",saved);
    const btn=document.getElementById("theme-btn");
    if(btn)btn.textContent=saved==="dark"?"☀️":"🌙";
});
"""

with gr.Blocks(css=css, title="Grounding DINO Object Detector") as demo:

    gr.HTML(f"""
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">🌙</button>
    <script>{js}</script>
    <div class="app-header">
        <h1>Grounding DINO Object Detector</h1>
        <p>Open-vocabulary object detection — detect anything you can describe in text.
        No fixed class limit. Works on galaxies, appliances, rooms, nature and more.</p>
        <div class="badges">
            <span class="badge accent">Grounding DINO</span>
            <span class="badge accent">232M Parameters</span>
            <span class="badge positive">Open Vocabulary</span>
            <span class="badge positive">Zero-shot</span>
            <span class="badge">Any Object</span>
            <span class="badge">8 Presets</span>
        </div>
    </div>
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">232M</div>
            <div class="stat-label">Parameters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">Unlimited</div>
            <div class="stat-label">Object Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">Zero-shot</div>
            <div class="stat-label">No Fine-tuning</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">8</div>
            <div class="stat-label">Quick Presets</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("### Upload Image")
            input_image = gr.Image(type="pil", label="Image", height=280)

            labels_input = gr.Textbox(
                label="Objects to Detect (comma separated)",
                placeholder="person, car, building, tree...",
                value="person, car, bus, building, tree",
                lines=2
            )

            threshold = gr.Slider(
                minimum=0.1, maximum=0.7,
                value=0.3, step=0.05,
                label="Detection Threshold"
            )

            detect_btn = gr.Button("Detect Objects", variant="primary")

            gr.Markdown("#### Quick Presets")
            with gr.Row():
                for name in list(presets.keys())[:4]:
                    btn = gr.Button(name, size="sm")
                    btn.click(fn=lambda x=presets[name]: x, outputs=labels_input)
            with gr.Row():
                for name in list(presets.keys())[4:]:
                    btn = gr.Button(name, size="sm")
                    btn.click(fn=lambda x=presets[name]: x, outputs=labels_input)

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("### Detection Results")
            output_image = gr.Image(label="Annotated Image", height=280)
            output_text  = gr.Markdown(value="Upload an image and click Detect Objects.")

    gr.HTML("""
    <div style="margin-top:28px;background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:28px;box-shadow:var(--shadow);">
        <h3 style="color:var(--text-primary);margin-bottom:20px;font-size:1em;font-weight:700;">Why Grounding DINO vs YOLOv8</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;">
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;">
                <div style="font-weight:700;color:var(--text-primary);font-size:0.9em;margin-bottom:8px;">YOLOv8</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.8;">
                    Fixed 80 COCO classes only<br>
                    Cannot detect custom objects<br>
                    Fast inference<br>
                    Struggles with rare objects
                </div>
            </div>
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;border:1px solid rgba(245,158,11,0.3);">
                <div style="font-weight:700;color:var(--accent);font-size:0.9em;margin-bottom:8px;">Grounding DINO</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.8;">
                    Unlimited custom classes<br>
                    Detect anything you describe<br>
                    Open-vocabulary detection<br>
                    State of the art accuracy
                </div>
            </div>
        </div>
    </div>
    <div class="footer">
        Grounding DINO Base • 232M Parameters • Open-vocabulary Detection •
        <a href="https://github.com/Boatengs/yolo-object-detector" target="_blank" style="color:var(--accent);">GitHub</a>
    </div>
    """)

    detect_btn.click(
        fn=detect_objects,
        inputs=[input_image, labels_input, threshold],
        outputs=[output_image, output_text]
    )

demo.launch()
