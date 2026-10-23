from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import time
from recognition_logic import train_lbph_model, mark_attendance, get_today_attendance, init_env

app = Flask(__name__)

# Basic Setup
init_env()
FACES_DIR = os.path.join(os.path.dirname(__file__), "static", "faces")
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Global variables for OpenCV LBPH Recognizer
recognizer = None
id_to_name = {}

# Global state to communicate with frontend
last_recognition = {
    "name": "unknown",
    "status": "waiting", # "recognized", "unknown", "waiting"
    "timestamp": 0
}

def reload_faces():
    global recognizer, id_to_name
    recognizer, id_to_name = train_lbph_model()
    print(f"Loaded {len(id_to_name)} distinct profiles into LBPH Engine.")

# Load faces on startup
reload_faces()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance_feed')
def attendance_feed():
    return render_template('attendance.html')

@app.route('/manage')
def manage():
    return render_template('manage.html')

@app.route('/register')
def register():
    return render_template('register.html')

# --- API ENDPOINTS ---

@app.route('/api/today_logs')
def get_logs():
    return jsonify(get_today_attendance())

@app.route('/api/latest_recognition')
def get_latest_recognition():
    """Endpoint for frontend to poll for audio/visual feedback"""
    global last_recognition
    return jsonify(last_recognition)

@app.route('/api/reload_faces', methods=['POST'])
def api_reload_faces():
    reload_faces()
    return jsonify({"success": True, "message": "Faces reloaded."})

# --- VIDEO STREAMING LOGIC ---

def generate_frames():
    global last_recognition, recognizer, id_to_name
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Convert to grayscale for Haar Cascade and LBPH
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            name = "Unknown"
            
            # Predict the face
            if recognizer is not None:
                try:
                    id_, confidence = recognizer.predict(roi_gray)
                    # For LBPH, a lower confidence value means a smaller distance (better match).
                    # A threshold of ~70-85 is typical for a decent match.
                    if confidence < 80:
                        name = id_to_name.get(id_, "Unknown")
                except Exception as e:
                    pass
            
            # Update global state for audio & attendance logs
            current_time = time.time()
            if current_time - last_recognition['timestamp'] > 5 or last_recognition['name'] != name:
                if name != "Unknown":
                    mark_attendance(name)
                    last_recognition = {"name": name, "status": "recognized", "timestamp": current_time}
                else:
                    last_recognition = {"name": "Unknown", "status": "unrecognized", "timestamp": current_time}

            # Draw Visual Bounding Box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y - 6), font, 0.8, (255, 255, 255), 1)

        # Encode frame back to JPEG for the stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# --- FACE MANAGEMENT APIS ---

import base64
@app.route('/api/register_face', methods=['POST'])
def register_face():
    data = request.json
    name = data.get('name')
    image_data = data.get('image') # Base64 string
    
    if not name or not image_data:
        return jsonify({"success": False, "error": "Missing name or image."}), 400
        
    person_dir = os.path.join(FACES_DIR, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        
    # generate unique filename
    filename = f"{int(time.time())}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    # decode base64
    header, encoded = image_data.split(",", 1)
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(encoded))
        
    reload_faces()
    return jsonify({"success": True, "message": "Face registered successfully."})

@app.route('/api/get_registered_faces')
def get_registered_faces():
    faces_data = []
    if os.path.exists(FACES_DIR):
        for name in os.listdir(FACES_DIR):
            person_dir = os.path.join(FACES_DIR, name)
            if os.path.isdir(person_dir):
                count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                faces_data.append({"name": name, "image_count": count})
    return jsonify(faces_data)

@app.route('/api/delete_face', methods=['POST'])
def delete_face():
    data = request.json
    name = data.get('name')
    person_dir = os.path.join(FACES_DIR, name)
    if os.path.exists(person_dir):
        import shutil
        shutil.rmtree(person_dir)
        reload_faces()
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
