import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import sys
from download_dataset import setup_dataset

# Create model
def build_emotion_model():
    """Build CNN model for emotion detection"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    return model

def load_csv_dataset(csv_file='fer2013.csv'):
    """Load FER2013 dataset from CSV file
    
    CSV should have columns: emotion, pixels, Usage
    emotion: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise
    pixels: space-separated pixel values (48x48=2304 pixels)
    Usage: 'Training' or 'PrivateTest'
    """
    print(f"[v0] Loading CSV dataset from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"[v0] ERROR: {csv_file} not found!")
        print("[v0] Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("[v0] Place fer2013.csv in the project root directory")
        return None, None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"[v0] Loaded {len(df)} samples from CSV")
        
        # Extract emotions and pixel data
        emotions = df['emotion'].values
        
        # Convert pixel string to image array
        images = []
        for pixels in df['pixels']:
            pixel_array = np.array(pixels.split(), dtype=np.uint8)
            image_reshaped = pixel_array.reshape(48, 48, 1)
            images.append(image_reshaped)
        
        images = np.array(images)
        print(f"[v0] Processed {len(images)} images")
        
        # Convert emotions to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        emotions_categorical = to_categorical(emotions, 7)
        
        return images, emotions_categorical
        
    except Exception as e:
        print(f"[v0] Error loading CSV: {e}")
        return None, None

def train_model_from_csv():
    """Train emotion detection model from CSV file"""
    print("[v0] Emotion Detection Model Training")
    print("[v0] ================================\n")
    
    # Try to load from CSV first
    X, y = load_csv_dataset('fer2013.csv')
    
    if X is None:
        print("[v0] CSV dataset not found. Setting up sample dataset...")
        setup_dataset()
        print("[v0] Using sample dataset for training...")
        return train_model_from_folders()
    
    print("[v0] Building emotion detection model...")
    model = build_emotion_model()
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    print("[v0] Model compiled successfully\n")
    
    # Split data into training and validation (use Usage column if available)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize pixel values to 0-1 range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"[v0] Training set: {len(X_train)} samples")
    print(f"[v0] Validation set: {len(X_test)} samples\n")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("[v0] Starting training...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
        steps_per_epoch=len(X_train) // 32
    )
    
    model.save('face_emotionModel.h5')
    print("\n[v0] Model saved as face_emotionModel.h5")
    print("[v0] Training complete!")

def train_model_from_folders():
    """Train emotion detection model from train/ and test/ folder structure
    
    Expected folder structure:
    train/
        ├── angry/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
    
    test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
    """
    print("[v0] Emotion Detection Model Training - From Folders")
    print("[v0] ================================================\n")
    
    # Check if train and test folders exist
    if not os.path.exists('train') or not os.path.exists('test'):
        print("[v0] ERROR: Missing folder structure!")
        print("[v0] Create 'train/' and 'test/' directories with emotion subfolders:")
        print("[v0]   train/angry/, train/disgust/, train/fear/, train/happy/, train/neutral/, train/sad/, train/surprise/")
        print("[v0]   test/angry/, test/disgust/, test/fear/, test/happy/, test/neutral/, test/sad/, test/surprise/")
        print("[v0] Place grayscale face images in respective emotion folders")
        return False
    
    print("[v0] Building emotion detection model...")
    model = build_emotion_model()
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        print("[v0] Loading training data from train/ folder...")
        train_data = train_datagen.flow_from_directory(
            'train',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        print("[v0] Loading test data from test/ folder...")
        test_data = test_datagen.flow_from_directory(
            'test',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"[v0] Training samples: {train_data.samples}")
        print(f"[v0] Test samples: {test_data.samples}")
        print(f"[v0] Classes: {train_data.class_indices}\n")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        print("[v0] Starting training...")
        history = model.fit(
            train_data,
            epochs=50,
            validation_data=test_data,
            callbacks=[early_stop],
            verbose=1
        )
        
        model.save('face_emotionModel.h5')
        print("\n[v0] Model saved as face_emotionModel.h5")
        print("[v0] Training complete!")
        return True
        
    except Exception as e:
        print(f"[v0] Error during training: {e}")
        print("[v0] Ensure train/ and test/ folders have correct structure with emotion subfolders")
        return False

if __name__ == "__main__":
    success = train_model_from_folders()
    
    if not success:
        print("\n[v0] Attempting fallback: training from CSV...")
        train_model_from_csv()
