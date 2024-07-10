import cv2
from ultralytics import YOLO
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# Initialize Firebase
try:
    cred = credentials.Certificate('C:/Users/acer/py/app/path/cardata-5ff57-firebase-adminsdk-ba8a0-5f85c53755.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://cardata-5ff57-default-rtdb.firebaseio.com'
    })
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit()

# Load YOLOv8 model
model = YOLO('yolov8.pt')  # Adjust the model path as needed

# Define parking blocks with coordinates (x, y, width, height)
parking_blocks = [
    (130, 15, 90, 170),  # Block 1
    (230, 15, 90, 170),  # Block 2
    (330, 15, 90, 170),  # Block 3
    (430, 15, 90, 170),  # Block 4
    (530, 15, 90, 170),  # Block 5
    (130, 270, 90, 170),  # Block 6
    (230, 270, 90, 170),  # Block 7
    (330, 270, 90, 170),  # Block 8
    (430, 270, 90, 170),  # Block 9
    (530, 270, 90, 170),  # Block 10
]

# Function to check if a car is in a block
def is_car_in_block(block, car_bbox):
    bx, by, bw, bh = block
    cx1, cy1, cx2, cy2 = car_bbox
    return (cx1 < bx + bw and cx2 > bx and cy1 < by + bh and cy2 > by)

# Function to detect cars and draw bounding boxes
def detect_car(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 320))
    results = model(small_frame, conf=0.4, iou=0.3)  # Adjust conf and iou for better accuracy

    occupied_blocks = set()
    for result in results:
        for bbox in result.boxes:
            class_id = int(bbox.cls)
            object_label = model.names[class_id]

            # Convert bounding box coordinates back to the original frame size
            x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
            x1 = int(x1 * (frame.shape[1] / small_frame.shape[1]))
            y1 = int(y1 * (frame.shape[0] / small_frame.shape[0]))
            x2 = int(x2 * (frame.shape[1] / small_frame.shape[1]))
            y2 = int(y2 * (frame.shape[0] / small_frame.shape[0]))
            car_bbox = (x1, y1, x2, y2)

            # Check which block the car is in
            for i, block in enumerate(parking_blocks):
                if is_car_in_block(block, car_bbox):
                    occupied_blocks.add(i)

            # Display object label on frame
            cv2.putText(frame, object_label, (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw parking blocks and display available spaces
    for i, (x, y, w, h) in enumerate(parking_blocks):
        color = (0, 0, 255) if i in occupied_blocks else (0, 255, 0)  # Red if occupied, green if available
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"P {i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    available_spaces = len(parking_blocks) - len(occupied_blocks)
    cv2.putText(frame, f"Available Spaces: {available_spaces}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Optionally, save the data to Firebase
    db.reference('/parking').set({
        'available_spaces': available_spaces,
        'occupied_blocks': list(occupied_blocks),
        'timestamp': datetime.now().isoformat()
    })

    return frame

# Initialize the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_car(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame', frame_bytes)
        socketio.sleep(0.1)  # Adjust the sleep time to control the frame rate

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(target=generate_frames)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    finally:
        cap.release()
