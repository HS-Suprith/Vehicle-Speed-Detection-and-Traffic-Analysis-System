# test.py (Final Refactored Version)

import cv2
from ultralytics import YOLO
from tracker import Tracker

# --- SETTINGS / CONFIGURATIONS ---
VIDEO_PATH = 'rom.mp4'
MODEL_PATH = 'yolov8s.pt'
COCO_CLASSES_PATH = 'coco.txt'
CLASS_TO_TRACK = 'car'
FRAME_WIDTH = 1020
FRAME_HEIGHT = 500

# Counting Line Configuration
LINE_Y_POSITION = 322
LINE_OFFSET = 6
# --- END OF SETTINGS ---

def main():
    """
    Main function to run the vehicle tracking and counting.
    """
    # Load the YOLO model
    model = YOLO(MODEL_PATH)

    # Load class names from coco.txt
    with open(COCO_CLASSES_PATH, "r") as f:
        class_list = f.read().split("\n")

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Initialize tracker and counting variables
    tracker = Tracker()
    vehicle_counter = 0
    vehicles_passed = set()

    # Main loop for processing video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Resize the frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Perform object detection
        results = model.predict(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                class_name = class_list[int(class_id)]

                if CLASS_TO_TRACK in class_name:
                    w, h = x2 - x1, y2 - y1
                    detections.append([int(x1), int(y1), int(w), int(h)])

        # Update tracker with detections
        tracked_objects = tracker.update(detections)

        # Draw counting line
        cv2.line(frame, (0, LINE_Y_POSITION), (FRAME_WIDTH, LINE_Y_POSITION), (255, 255, 0), 2)

        # Process tracked objects for counting
        for obj in tracked_objects:
            x, y, w, h, obj_id = obj
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate center point
            center_y = y + h // 2

            # Check for line crossing
            if LINE_Y_POSITION - LINE_OFFSET < center_y < LINE_Y_POSITION + LINE_OFFSET:
                if obj_id not in vehicles_passed:
                    vehicle_counter += 1
                    vehicles_passed.add(obj_id)

        # Display the vehicle count
        cv2.putText(frame, f"{CLASS_TO_TRACK.capitalize()} Count: {vehicle_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Vehicle Tracking and Counting", frame)

        # Exit on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()