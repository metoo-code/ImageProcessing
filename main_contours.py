import cv2
import base64
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template
from flask_socketio import SocketIO

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

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize slot_empty list
num_boxes = 12
slot_empty = [True] * num_boxes

def detect_car(frame):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_width = 60
    box_height = 100
    gap = 27
    empty_slots = num_boxes

    for i in range(num_boxes):
        # Calculate x and y positions for each box
        if i < 6:
            # First row
            x = i * (box_width + gap) + 107
            y = 55
        else:
            # Second row
            x = (i - 6) * (box_width + gap) + 107
            y = 200 + box_height + gap

        # Assume the slot is empty initially
        slot_empty[i] = True
        edge_detected = False

        for contour in contours:
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)

            if (x < x_contour < x + box_width and y < y_contour < y + box_height and
                    x < x_contour + w_contour < x + box_width and y < y_contour + h_contour < y + box_height):
                if (x_contour <= x or x_contour + w_contour >= x + box_width or
                        y_contour <= y or y_contour + h_contour >= y + box_height):
                    edge_detected = True
                slot_empty[i] = False
                break

        # Draw rectangle and update the slot status
        if slot_empty[i]:
            color = (0, 255, 0)  # Green if empty
        else:
            if edge_detected:
                # Blink the border if an object is detected on the edge
                if int(datetime.now().timestamp() * 10) % 2 == 0:
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Yellow
            else:
                color = (0, 0, 255)  # Red if occupied

        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)

        # Display sequence number of the box
        cv2.putText(frame, f"{i + 1}", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Count empty slots
    empty_slots = sum(slot_empty)

    # Write text to display the number of empty slots
    cv2.putText(frame, f"free: {empty_slots}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Write text to display the current date and time
    current_time = datetime.now().strftime(" %Y-%m-%d %H:%M:%S ")
    cv2.putText(frame, current_time, (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    return frame


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
