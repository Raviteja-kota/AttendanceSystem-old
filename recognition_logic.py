import os
import cv2
import numpy as np
from datetime import datetime
import csv

FACES_DIR = os.path.join(os.path.dirname(__file__), "static", "faces")
ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), "attendance.csv")

def init_env():
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time", "Date"])

def train_lbph_model():
    """
    Scans the faces directory, detects faces using Haar Cascade,
    and trains an LBPH Face Recognizer model.
    Returns: Tuple (model, id_to_name_dict)
    """
    faces = []
    ids = []
    id_to_name = {}
    current_id = 0
    
    if not os.path.exists(FACES_DIR):
        return None, {}
        
    # Load OpenCV Haar cascade for face detection
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    for name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, name)
        if os.path.isdir(person_dir):
            current_id += 1
            id_to_name[current_id] = name
            
            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, filename)
                    try:
                        # Load image in grayscale
                        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img_gray is None: continue
                        
                        # Detect face bounding box (robustness measure)
                        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
                        
                        # If a face is found, append it
                        for (x, y, w, h) in detected_faces:
                            face_roi = img_gray[y:y+h, x:x+w]
                            faces.append(face_roi)
                            ids.append(current_id)
                            break # Just use the first detected face per image
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        
    if len(faces) == 0:
        return None, {}
        
    # Create and train LBPH model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(ids))
    return model, id_to_name

def mark_attendance(name):
    """
    Marks attendance uniquely per person per day.
    """
    init_env()
    data = []
    with open(ATTENDANCE_FILE, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    now = datetime.now()
    dString = now.strftime('%Y-%m-%d')
    tString = now.strftime('%H:%M:%S')
    
    already_marked = False
    for row in data[1:]: # skip header
        if len(row) >= 3:
            r_name, r_time, r_date = row
            if r_name == name and r_date == dString:
                already_marked = True
                break
                
    if not already_marked:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, tString, dString])
        return True # Added today
    return False # Already existed today

def get_today_attendance():
    init_env()
    logs = []
    try:
        now = datetime.now()
        dString = now.strftime('%Y-%m-%d')
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            # Iterate in reverse for newest first
            for row in reversed(data[1:]):
                if len(row) >= 3:
                    if row[2] == dString:
                        logs.append({"name": row[0], "time": row[1], "date": row[2]})
    except Exception as e:
        pass
    return logs
