# config.py

# File Paths (Updated for new folder structure)
VIDEO_PATH = '../data/rom.mp4'
MODEL_PATH = '../models/yolov8s.pt'
COCO_CLASSES_PATH = '../data/coco.txt'
DB_PATH = 'traffic_data.db' # Yeh src folder mein banegi

# Classes to track and count
CLASSES_TO_TRACK = ['car', 'truck', 'bus', 'motorcycle']

# Frame dimensions
FRAME_WIDTH = 1020
FRAME_HEIGHT = 500

# Speed estimation settings
PIXELS_PER_METER = 20

# Traffic density thresholds
DENSITY_LOW_THRESHOLD = 10
DENSITY_HIGH_THRESHOLD = 20

# Abnormal stop detection settings (seconds)
STOP_THRESHOLD_SECONDS = 5