# Vehicle Speed Detection and Traffic Analysis System

This project detects vehicle speed and analyzes traffic using computer vision
and deep learning techniques.

## Features
- Vehicle detection and tracking
- Speed estimation
- Traffic analysis and visualization

## Tech Stack
- Python
- OpenCV
- YOLOv8

## Model Download
The YOLOv8 model file is **not included** in this repository.

It will be downloaded automatically at runtime:
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
