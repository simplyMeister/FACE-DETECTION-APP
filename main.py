import os
import numpy as np
from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained emotion detection model
MODEL_PATH = "face_emotionModel.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load model if it exists
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"[v0] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[v0] Error loading model: {e}")
else:
    print(f"[v0] Warning: {MODEL_PATH} not found. Predictions will fail.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Detect emotions in uploaded image"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 503
        
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Read image
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({"emotion": "No face detected", "confidence": 0}), 200
        
        # Process first detected face
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to model input size (48x48)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype("float32") / 255.0
        roi_input = np.expand_dims(np.expand_dims(roi_resized, -1), 0)
        
        # Predict emotion
        prediction = model.predict(roi_input, verbose=0)
        emotion_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][emotion_idx])
        
        # Return results with all emotion confidences
        result = {
            "emotion": emotion_labels[emotion_idx],
            "confidence": round(confidence * 100, 2),
            "all_emotions": {
                emotion_labels[i]: round(float(prediction[0][i]) * 100, 2)
                for i in range(len(emotion_labels))
            }
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def stats():
    """Return emotion statistics (placeholder)"""
    return jsonify({
        "total_predictions": 0,
        "emotions": {label: 0 for label in emotion_labels}
    }), 200

@app.route("/history", methods=["GET"])
def history():
    """Return prediction history (placeholder)"""
    return jsonify([]), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)