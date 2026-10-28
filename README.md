# Open Vocabulary Object Detector (Grounding DINO)

Type the objects you want to find, upload an image, and the app draws a box around each match with a confidence score. It runs Grounding DINO, so it isn't limited to a fixed list of classes.

This project started as a YOLOv8 detector. I replaced YOLOv8 with Grounding DINO after testing showed the limits of a closed set of 80 COCO classes (see [Version 1](#version-1-yolov8) below). The repository name still says YOLO so existing links keep working.

## Live demo

https://huggingface.co/spaces/samurvivor-07/yolov8-object-detector

## How it works

The app loads `IDEA-Research/grounding-dino-base` through Hugging Face Transformers and runs on GPU when one is available.

1. You enter a comma separated list of objects, for example `person, car, bus, building, tree`.
2. The list is turned into a text prompt (`person . car . bus . building . tree .`) and passed to the model with the image.
3. Detections above the threshold you set are drawn on the image, and a summary counts how many of each object were found.

No fine tuning is involved. Any label you can describe in plain text is a valid query.

## Features

* Free text labels, so there's no fixed class list
* Detection threshold slider from 0.1 to 0.7 (default 0.3)
* Eight label presets: Street Scene, Kitchen, Living Room, Space & Galaxy, Animals, Sports, Office, Nature
* Annotated output image with a per label count
* Light and dark theme

## Version 1: YOLOv8

The first version used YOLOv8 Large trained on COCO. These are the results I recorded from that version; the screenshots in this repo come from it.

| Scene | Objects detected | Top confidence |
|-------|-----------------|----------------|
| Street scene | 5 (4 people, 1 bus) | 95% bus |
| Kitchen scene | 9 (bowls, people, oven, apple) | 96% person |
| Sports scene | 4 (2 people, 2 ties) | 94% person |
| Wildlife (fox) | 2 (cat, dog) | 69% cat |

The wildlife result is why I switched models. COCO has no fox, rabbit, squirrel or deer, so YOLOv8 mapped them to the closest class it knew: a fox came back as a cat and a dog, a rabbit as a cat, and a squirrel as a bear. A closed set detector can't say "I don't know this object." It forces a match. Open vocabulary models like Grounding DINO address that by letting the user name the target.

## Tech stack

* Grounding DINO (`IDEA-Research/grounding-dino-base`) via Hugging Face Transformers
* PyTorch
* Gradio
* Hugging Face Spaces
* Google Colab T4 GPU for development

## Run locally

```bash
git clone https://github.com/Boatengs/yolo-object-detector.git
cd yolo-object-detector
pip install -r requirements.txt
python app.py
```

The model weights download from the Hugging Face Hub on first run.

## Files

* `app.py`: the Grounding DINO Gradio app
* `requirements.txt`: dependencies for the app
* `yolo_object_detection.ipynb`: development notebook covering the YOLOv8 version and the Grounding DINO upgrade
* `notebooks/99_review_walkthrough.ipynb`: reviewer walkthrough
* `Street_Scene_Detection.png`, `Kitchen_Scene_Detection.png`, `Sports_Scene_Detection.png`, `Wildlife_Detection.png`: YOLOv8 test results

## Portfolio links

* Project 8 CLIP Vision Language: https://github.com/Boatengs/clip-vision-language
* Project 7 Skin Lesion Segmentation: https://github.com/Boatengs/skin-lesion-segmentation
* Project 5 Skin Disease Classifier: https://github.com/Boatengs/skin-disease-classifier
* Project 4 LLM Eval Framework: https://github.com/Boatengs/llm-evaluation-framework
* Project 3 Medical QA LoRA: https://github.com/Boatengs/medical-qa-lora
* Project 2 SPORTZBOT RAG: https://github.com/Boatengs/sports-rag-chatbot-
* Project 1 Sentiment Analyzer: https://github.com/Boatengs/sentiment-analyzer
