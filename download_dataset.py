"""
Download and prepare FER2013 dataset automatically
Handles CSV parsing and image organization
"""
import os
import csv
import numpy as np
from PIL import Image
import requests
import zipfile
import io

def download_fer2013_csv():
    """Download FER2013 CSV from kaggle or alternative source"""
    print("[v0] Downloading FER2013 dataset...")
    
    # Try downloading from a public source (if available)
    # Alternative: Use Kaggle API (requires API credentials)
    url = "https://www.dropbox.com/s/opuvvkeq80744fn/fer2013.csv?dl=1"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            print("[v0] Dataset downloaded successfully")
            return response.content.decode('utf-8')
    except Exception as e:
        print(f"[v0] Could not download from public source: {e}")
    
    return None

def create_directories():
    """Create dataset directories"""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    for split in ['train', 'test']:
        for emotion in emotions:
            path = os.path.join(split, emotion)
            os.makedirs(path, exist_ok=True)
    
    print("[v0] Created dataset directories")

def parse_fer2013_csv(csv_content):
    """Parse FER2013 CSV and save images"""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    if not csv_content:
        print("[v0] CSV content not available")
        return False
    
    try:
        lines = csv_content.strip().split('\n')
        count = 0
        
        for idx, line in enumerate(lines[1:]):  # Skip header
            if idx % 1000 == 0:
                print(f"[v0] Processing image {idx}...")
            
            parts = line.split(',')
            if len(parts) >= 3:
                emotion = int(parts[0])
                pixels = parts[1]
                usage = parts[2].strip()
                
                # Create image from pixel data
                pixel_array = np.array(pixels.split(), dtype=np.uint8)
                pixel_array = pixel_array.reshape(48, 48)
                
                # Convert to PIL Image
                img = Image.fromarray(pixel_array, mode='L')
                
                # Determine split (train/test)
                split = 'train' if usage == 'Training' else 'test'
                
                # Save image
                emotion_name = emotions[emotion]
                filename = f"{split}_{idx}.png"
                filepath = os.path.join(split, emotion_name, filename)
                img.save(filepath)
                
                count += 1
                if count >= 5000:  # Limit for initial setup (can increase)
                    break
        
        print(f"[v0] Processed {count} images")
        return count > 0
        
    except Exception as e:
        print(f"[v0] Error parsing CSV: {e}")
        return False

def create_sample_dataset():
    """Create sample dataset with synthetic images for testing"""
    print("[v0] Creating sample dataset for testing...")
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    create_directories()
    
    # Create 10 sample images per emotion
    for split in ['train', 'test']:
        for emotion_idx, emotion in enumerate(emotions):
            for i in range(10):
                # Create random 48x48 grayscale image
                pixel_data = np.random.randint(50, 200, (48, 48), dtype=np.uint8)
                
                # Add pattern based on emotion for variety
                if emotion == 'Happy':
                    pixel_data[20:28, 15:33] = 150
                elif emotion == 'Sad':
                    pixel_data[15:25, 15:33] = 100
                elif emotion == 'Angry':
                    pixel_data[15:20, 10:38] = 180
                
                img = Image.fromarray(pixel_data, mode='L')
                filename = f"{split}_{emotion_idx}_{i}.png"
                filepath = os.path.join(split, emotion, filename)
                img.save(filepath)
    
    print("[v0] Sample dataset created with 10 images per emotion")

def setup_dataset():
    """Main setup function"""
    print("[v0] Starting dataset setup...")
    
    # Check if dataset already exists
    if os.path.exists('train') and len(os.listdir('train')) > 0:
        print("[v0] Dataset already exists, skipping download")
        return True
    
    # Try to download real dataset
    create_directories()
    csv_content = download_fer2013_csv()
    
    if csv_content and parse_fer2013_csv(csv_content):
        print("[v0] FER2013 dataset setup complete")
        return True
    else:
        print("[v0] Using sample dataset for testing")
        create_sample_dataset()
        return True

if __name__ == "__main__":
    setup_dataset()
