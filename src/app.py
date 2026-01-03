# app.py (Final Consolidated Version with All Features)

import cv2
import streamlit as st
from ultralytics import YOLO
from tracker import Tracker
import pandas as pd
from datetime import datetime
import math
import numpy as np
import sqlite3
from pathlib import Path

# --- PATHS & SETTINGS ---
# Get the directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Define paths for all external files relative to the script's location
# IMPORTANT: Make sure these files are in the same folder as this script
VIDEO_PATH = BASE_DIR / 'test2.mp4'
MODEL_PATH = BASE_DIR / 'yolov8s.pt'
COCO_CLASSES_PATH = BASE_DIR / 'coco.txt'
DB_PATH = BASE_DIR / 'traffic_data.db'

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
# --- END OF SETTINGS ---

# --- DATABASE SETUP ---
def init_db():
    """Initializes the database and creates the 'detections' table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                vehicle_id INTEGER,
                vehicle_type TEXT,
                speed_kmh REAL,
                direction TEXT,
                event TEXT
            )
        ''')
        conn.commit()

def add_detection_to_db(timestamp, vehicle_id, vehicle_type, speed_kmh, direction, event="Line Cross"):
    """Adds a single detection event to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (timestamp, vehicle_id, vehicle_type, speed_kmh, direction, event) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, vehicle_id, vehicle_type, speed_kmh, direction, event)
        )
        conn.commit()

# Initialize the database and table at the start of the script
init_db()
# --- END OF DATABASE SETUP ---

# Initialize session state variables
if 'run' not in st.session_state:
    st.session_state.run = False

def start_processing():
    """Callback function for the 'Start' button."""
    st.session_state.run = True
    # Reset data for a new run
    st.session_state.csv_data = []
    st.session_state.chart_data = pd.DataFrame(columns=['Frame'] + CLASSES_TO_TRACK)
    st.session_state.illegal_stops = set()

def main():
    st.set_page_config(page_title="RoadGuardian AI Ultimate", page_icon="🏆", layout="wide")
    st.title("Vehicle Speed Detection and Traffic Analysis System🏆")
    st.write("Full-featured traffic monitoring with multi-class counting, speed/direction estimation, density analysis, stop detection, and heatmap generation.")

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Configuration")
    line_y_position = st.sidebar.slider(
        'Set Counting Line Position (ROI)', min_value=0, max_value=FRAME_HEIGHT, value=340)
    st.sidebar.button('Start / Restart Analysis', on_click=start_processing, type="primary")

    # --- MAIN VIDEO AND DASHBOARD AREA ---
    col1, col2 = st.columns([3, 2])
    with col1:
        st_frame = st.empty()
        chart_placeholder = st.empty()
    with col2:
        st.subheader("📊 Live Dashboard")
        density_placeholder = st.empty()
        st.markdown("---")
        counts_placeholders = {cls: st.empty() for cls in CLASSES_TO_TRACK}
        total_count_placeholder = st.empty()

    if not st.session_state.run:
        st.info("Click 'Start / Restart Analysis' in the sidebar to begin.")
        return

    # --- ROBUSTNESS CHECKS FOR ESSENTIAL FILES ---
    essential_files = {'Model': MODEL_PATH, 'Classes': COCO_CLASSES_PATH, 'Video': VIDEO_PATH}
    for name, path in essential_files.items():
        if not path.exists():
            st.error(f"Error: {name} file not found at '{path}'. Please check the file location.")
            st.session_state.run = False
            return

    # Load model and class names
    model = YOLO(MODEL_PATH)
    with open(COCO_CLASSES_PATH, "r") as f:
        class_list = f.read().strip().split("\n")

    # Open video file and check if it's valid
    cap = cv2.VideoCapture(str(VIDEO_PATH)) # Use str() for compatibility
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at '{VIDEO_PATH}'. It may be corrupted or in an unsupported format.")
        st.session_state.run = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize tracker and data structures
    tracker = Tracker()
    vehicle_counts = {cls: 0 for cls in CLASSES_TO_TRACK}
    vehicles_passed = {cls: set() for cls in CLASSES_TO_TRACK}
    object_info = {}
    frame_count = 0
    heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.success("Video processing complete! You can now export the data.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1
        
        results = model.predict(frame, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = box
                if int(class_id) < len(class_list) and class_list[int(class_id)] in CLASSES_TO_TRACK:
                    detections.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), class_list[int(class_id)]])

        tracked_objects = tracker.update([d[:4] for d in detections])
        
        current_frame_detections = len(tracked_objects)
        for obj in tracked_objects:
            x, y, w, h, obj_id = obj
            center_x, center_y = x + w // 2, y + h // 2
            heatmap[y:y+h, x:x+w] += 1.0 # Update heatmap over the whole box for better viz

            if obj_id not in object_info:
                # Naive association of tracker ID with detection class name
                for det in detections:
                    if abs(x - det[0]) < 20 and abs(y - det[1]) < 20:
                        object_info[obj_id] = {'class': det[4], 'positions': [], 'last_frame': 0, 'stopped_frames': 0}
                        break
            
            if obj_id in object_info:
                object_info[obj_id]['positions'].append((center_x, center_y))
                object_info[obj_id]['last_frame'] = frame_count
                class_name = object_info[obj_id]['class']

                speed_kmh, direction = 0, ""
                if len(object_info[obj_id]['positions']) > 1:
                    prev_pos, curr_pos = object_info[obj_id]['positions'][-2:]
                    direction = "Down" if curr_pos[1] > prev_pos[1] else "Up"
                    dist_pixels = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                    speed_kmh = (dist_pixels / PIXELS_PER_METER) * fps * 3.6
                
                # Abnormal Stop Detection
                is_stopped = speed_kmh < 1.0
                box_color = (0, 255, 0) # Green for moving
                if is_stopped:
                    object_info[obj_id]['stopped_frames'] += 1
                else:
                    object_info[obj_id]['stopped_frames'] = 0

                if object_info[obj_id]['stopped_frames'] > (fps * STOP_THRESHOLD_SECONDS):
                    box_color = (0, 0, 255) # Red for stopped
                    cv2.putText(frame, "ILLEGAL STOP", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    if obj_id not in st.session_state.illegal_stops:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        add_detection_to_db(timestamp, obj_id, class_name, speed_kmh, direction, "Illegal Stop")
                        st.session_state.illegal_stops.add(obj_id)

                # Drawing on Frame
                label = f"ID:{obj_id} {class_name.capitalize()}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"{speed_kmh:.1f} km/h {direction}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Multi-Class Counting
                if line_y_position - 6 < center_y < line_y_position + 6 and obj_id not in vehicles_passed.get(class_name, set()):
                    vehicle_counts[class_name] += 1
                    vehicles_passed.setdefault(class_name, set()).add(obj_id)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.csv_data.append([timestamp, obj_id, class_name, f"{speed_kmh:.1f}"])
                    add_detection_to_db(timestamp, obj_id, class_name, speed_kmh, direction, "Line Cross")

        # Update Dashboard
        density = current_frame_detections
        density_status = "Low" if density < DENSITY_LOW_THRESHOLD else "Medium" if density < DENSITY_HIGH_THRESHOLD else "High (Jam)"
        density_placeholder.metric("Current Traffic Density", f"{density_status} ({density} vehicles)")
        for cls in CLASSES_TO_TRACK:
            counts_placeholders[cls].metric(f"{cls.capitalize()} Count", vehicle_counts[cls])
        total_count_placeholder.metric("Total Vehicle Count", sum(vehicle_counts.values()))
        
        # Update Chart Data
        new_chart_data = {'Frame': frame_count, **vehicle_counts}
        st.session_state.chart_data = pd.concat([st.session_state.chart_data, pd.DataFrame([new_chart_data])], ignore_index=True)
        if frame_count % 10 == 0:
            chart_placeholder.line_chart(st.session_state.chart_data.set_index('Frame'))

        # Heatmap Overlay
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        superimposed_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        # Display Counting Line
        cv2.line(superimposed_frame, (0, line_y_position), (FRAME_WIDTH, line_y_position), (255, 255, 0), 2)
        
        # Display Final Frame
        st_frame.image(cv2.cvtColor(superimposed_frame, cv2.COLOR_BGR2RGB), caption='Live Analysis with Heatmap')

    # Data Export Button
    if st.session_state.csv_data:
        df_export = pd.DataFrame(st.session_state.csv_data, columns=['Timestamp', 'ID', 'Vehicle_Type', 'Speed_kmh'])
        st.sidebar.download_button(
            label="Export Data to CSV", data=df_export.to_csv(index=False).encode('utf-8'),
            file_name='traffic_analysis_report.csv', mime='text/csv')
    
    cap.release()

if __name__ == "__main__":
    main()