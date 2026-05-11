# ⚡ YOLOv8 Object Detector

Real-time object detection using YOLOv8 Large trained on COCO (80 classes).
Upload any image and detect objects instantly — people, vehicles, animals, food, furniture and more.

## Live Demo

https://huggingface.co/spaces/samboateng190/yolov8-object-detector

## What It Can Detect

80 COCO classes including:
- People and body parts
- Vehicles: car, bus, truck, motorcycle, bicycle, airplane, boat
- Animals: cat, dog, horse, cow, elephant, bear, zebra, giraffe
- Food: banana, apple, sandwich, pizza, donut, cake, hot dog
- Furniture: chair, couch, bed, dining table, toilet
- Electronics: laptop, phone, TV, keyboard, mouse, remote
- Kitchen: bottle, cup, fork, knife, spoon, bowl, microwave, oven

## Key Features

- YOLOv8 Large — 43.7M parameters, highest accuracy
- Adjustable confidence threshold (10-90%)
- 4 model sizes — nano (fastest) to large (most accurate)
- Works on any image type — street, sports, wildlife, kitchen, indoor
- Real-time bounding boxes with confidence scores
- Live demo on HuggingFace Spaces

## Test Results

| Scene | Objects Detected | Top Confidence |
|-------|-----------------|----------------|
| Street scene | 5 (4 people, 1 bus) | 95% bus |
| Kitchen scene | 9 (bowls, people, oven, apple) | 96% person |
| Sports scene | 4 (2 people, 2 ties) | 94% person |
| Wildlife (fox) | 2 (cat, dog) | 69% cat |

## Interesting Finding — Out-of-Distribution Detection

COCO does not include fox, rabbit, squirrel or deer.
When these animals appear YOLO maps them to the nearest known class:
- Fox detected as cat + dog
- Rabbit detected as cat
- Squirrel detected as bear

This demonstrates a key limitation of closed-set object detectors
and motivates open-vocabulary models like CLIP and OWL-ViT.

## Tech Stack

- YOLOv8 (Ultralytics)
- PyTorch
- Gradio
- OpenCV
- HuggingFace Spaces
- Google Colab T4 GPU

## Project Structure

- yolo_object_detection.ipynb — Full development notebook
- app.py — Gradio web application
- requirements.txt — Dependencies
- Street_Scene_Detection.png — Test result
- Kitchen_Scene_Detection.png — Test result
- Sports_Scene_Detection.png — Test result
- Wildlife_Detection.png — Test result

## Portfolio Links

- Project 5 Skin Disease Classifier: https://github.com/Boatengs/skin-disease-classifier
- Project 4 LLM Eval Framework: https://github.com/Boatengs/llm-evaluation-framework
- Project 3 Medical QA LoRA: https://github.com/Boatengs/medical-qa-lora
- Project 2 SPORTZBOT RAG: https://github.com/Boatengs/sports-rag-chatbot-
- Project 1 Sentiment Analyzer: https://github.com/Boatengs/sentiment-analyzer
