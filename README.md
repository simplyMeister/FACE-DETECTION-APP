# Facial Emotion Detection System

A Flask-based web application that detects emotions from facial images using deep learning.

## Features

- Real-time emotion detection from uploaded images
- 7 emotion classifications: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Confidence scores and probability distribution
- Prediction history tracking with SQLite database
- Statistics dashboard showing emotion trends
- Responsive web interface with modern UI
- Drag-and-drop image upload

## Project Structure

\`\`\`
├── app.py                    # Flask application & API routes
├── model_training.py         # CNN model training script
├── requirements.txt          # Python dependencies
├── database.db              # SQLite database (auto-created)
├── face_emotionModel.h5     # Trained emotion detection model
└── templates/
    └── index.html           # Frontend interface
\`\`\`

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone or download the project**
   \`\`\`bash
   cd facial-emotion-detection
   \`\`\`

2. **Create a virtual environment (recommended)**
   \`\`\`bash
   python -m venv venv
   \`\`\`

3. **Activate virtual environment**
   - Windows:
     \`\`\`bash
     venv\Scripts\activate
     \`\`\`
   - macOS/Linux:
     \`\`\`bash
     source venv/bin/activate
     \`\`\`

4. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

### Running the Application

1. **Start the Flask server**
   \`\`\`bash
   python app.py
   \`\`\`

2. **Open in browser**
   - Navigate to `http://localhost:5000`

3. **Upload an image**
   - Click the upload area or drag & drop an image
   - The system will detect faces and predict emotions
   - View confidence scores and emotion probabilities

### Training a Custom Model

If you want to train a new model with the FER2013 dataset:

1. **Download FER2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

2. **Organize the dataset**
   \`\`\`
   project_root/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       └── (same structure)
   \`\`\`

3. **Train the model**
   \`\`\`bash
   python model_training.py
   \`\`\`

   This will:
   - Build a CNN model
   - Load and preprocess images
   - Train for up to 50 epochs with early stopping
   - Save the model as `face_emotionModel.h5`

## API Endpoints

### POST `/predict`
Upload an image for emotion detection.

**Request:**
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
\`\`\`json
{
  "emotion": "Happy",
  "confidence": 85.23,
  "all_emotions": {
    "Angry": 2.1,
    "Disgust": 1.5,
    "Fear": 3.2,
    "Happy": 85.23,
    "Neutral": 5.8,
    "Sad": 1.9,
    "Surprise": 0.3
  }
}
\`\`\`

### GET `/history`
Retrieve prediction history (last 50).

**Response:**
\`\`\`json
[
  {
    "filename": "image_20240115_120530.jpg",
    "emotion": "Happy",
    "confidence": 85.23,
    "timestamp": "2024-01-15 12:05:30"
  }
]
\`\`\`

### GET `/stats`
Get emotion statistics from all predictions.

**Response:**
\`\`\`json
[
  {
    "emotion": "Happy",
    "count": 45,
    "avg_confidence": 78.5
  }
]
\`\`\`

## Deployment Options

### Option 1: Local Deployment
Perfect for personal/testing use.
- Run `python app.py`
- Access via `http://localhost:5000`

### Option 2: Render Deployment

1. **Create Render account** at [render.com](https://render.com)

2. **Create Web Service**
   - Connect GitHub repository
   - Select Python as runtime
   - Set start command: `gunicorn app:app`

3. **Add environment variables** (if needed)

4. **Deploy**
   - Render automatically detects `requirements.txt`
   - Your app will be live on a public URL

### Option 3: Heroku Deployment

1. **Install Heroku CLI** from [heroku.com](https://www.heroku.com/platform/cli)

2. **Create Procfile** in project root:
   \`\`\`
   web: gunicorn app:app
   \`\`\`

3. **Add gunicorn to requirements.txt**
   \`\`\`bash
   echo "gunicorn" >> requirements.txt
   \`\`\`

4. **Deploy**
   \`\`\`bash
   heroku login
   heroku create your-app-name
   git push heroku main
   \`\`\`

### Option 4: Docker Deployment

1. **Create Dockerfile**
   \`\`\`dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   \`\`\`

2. **Build and run**
   \`\`\`bash
   docker build -t emotion-detector .
   docker run -p 5000:5000 emotion-detector
   \`\`\`

## Model Information

### Architecture
- 4 Convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers with 512 units
- 7-class softmax output (emotions)

### Training Data
- FER2013 dataset (35,887 images)
- 48x48 grayscale images
- Balanced across 7 emotion categories

### Performance
- Validation accuracy: ~65-70% (typical)
- Real-time inference on CPU

## Troubleshooting

### Model not loading
- Ensure `face_emotionModel.h5` exists in project root
- If missing, run `python model_training.py` with dataset

### No face detected
- Ensure face is clearly visible in image
- Try different lighting/angle
- Recommended: face occupies at least 30% of image

### Port 5000 already in use
\`\`\`bash
# Change port in app.py (last line):
app.run(debug=True, port=5001)  # Use different port
\`\`\`

### Database errors
- Delete `database.db` to reset
- Restart the app (auto-creates new database)

## Tips for Best Results

1. **Image Quality**: Use clear, well-lit photos
2. **Face Visibility**: Ensure face is prominent and unobstructed
3. **File Size**: Keep images under 16MB
4. **Formats**: Supported formats are PNG and JPG

## License

This project uses pre-trained models and is for educational purposes.

## Future Enhancements

- Multiple face detection per image
- Real-time webcam emotion detection
- Emotion intensity analysis
- Custom model training UI
- Advanced analytics dashboard

## Support

For issues or questions, refer to the troubleshooting section or check the console for error messages.
