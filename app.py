from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import sqlite3
from datetime import datetime
import base64
from download_dataset import setup_dataset

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

setup_dataset()

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load trained emotion model
try:
    emotion_model = load_model('face_emotionModel.h5')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            face_image BLOB NOT NULL,
            face_x INTEGER,
            face_y INTEGER,
            face_w INTEGER,
            face_h INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[v0] Database initialized: database.db created/verified")

def save_face_to_db(filename, emotion, confidence, image_path, face_image_array, face_coords):
    """Save prediction with face image to database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    success, face_image_bytes = cv2.imencode('.png', face_image_array)
    if not success:
        face_image_bytes = b''
    else:
        face_image_bytes = face_image_bytes.tobytes()
    
    x, y, w, h = face_coords
    
    cursor.execute('''
        INSERT INTO predictions (filename, emotion, confidence, image_path, face_image, face_x, face_y, face_w, face_h)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (filename, emotion, confidence, image_path, face_image_bytes, x, y, w, h))
    
    conn.commit()
    stored_id = cursor.lastrowid
    conn.close()
    print(f"[v0] Face scanned and stored in database.db (ID: {stored_id}, Emotion: {emotion}, Confidence: {confidence}%)")

def detect_emotion(image_path):
    """Detect emotion from image and return face image"""
    if not model_loaded:
        return None, "Model not loaded"
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not read image"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, "No face detected"
        
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        emotion_pred = emotion_model.predict(roi_gray, verbose=0)
        emotion_idx = np.argmax(emotion_pred[0])
        emotion_label = EMOTIONS[emotion_idx]
        confidence = float(emotion_pred[0][emotion_idx]) * 100
        
        face_image = img[y:y+h, x:x+w]
        
        return {
            'emotion': emotion_label,
            'confidence': round(confidence, 2),
            'all_emotions': {
                EMOTIONS[i]: round(float(emotion_pred[0][i]) * 100, 2)
                for i in range(len(EMOTIONS))
            },
            'face_image': face_image,
            'face_coords': (x, y, w, h)
        }, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result, error = detect_emotion(filepath)
        
        if error:
            return jsonify({'error': error}), 400
        
        face_image = result.pop('face_image')
        face_coords = result.pop('face_coords')
        save_face_to_db(filename, result['emotion'], result['confidence'], filepath, face_image, face_coords)
        
        print(f"[v0] Prediction saved: {filename} - {result['emotion']} ({result['confidence']}%)")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[v0] Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, emotion, confidence, timestamp, face_image
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history_data = []
        for row in rows:
            face_b64 = ''
            if row[5]:
                face_b64 = base64.b64encode(row[5]).decode('utf-8')
            
            history_data.append({
                'id': row[0],
                'filename': row[1],
                'emotion': row[2],
                'confidence': row[3],
                'timestamp': row[4],
                'face_image': face_b64
            })
        
        return jsonify(history_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY emotion
            ORDER BY count DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        stats_data = [
            {
                'emotion': row[0],
                'count': row[1],
                'avg_confidence': round(row[2], 2)
            }
            for row in rows
        ]
        
        return jsonify(stats_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("[v0] ========================================")
    print("[v0] Facial Emotion Detection App Started")
    print("[v0] Database: database.db")
    print("[v0] Model Status: " + ('Loaded ✓' if model_loaded else 'Not Loaded ✗'))
    print("[v0] All scanned faces WILL be stored in database.db")
    print("[v0] ========================================")
    app.run(debug=True, port=5000)
