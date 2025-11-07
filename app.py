"""
Simple Flask Web App for Crop Disease Detection
Install: pip install flask
Run: python app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras # type: ignore
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
MODEL_PATH = 'final_best.keras'
model = keras.models.load_model(MODEL_PATH)
class_names = np.load('class_names.npy', allow_pickle=True)

# Disease info (simplified)
DISEASE_INFO = {
    'Tomato___Late_blight': {
        'name': 'Tomato Late Blight',
        'severity': 'High',
        'prevention': 'Use resistant varieties, apply fungicides, remove infected plants',
        'treatment': 'Copper-based fungicides, Mancozeb'
    },
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight',
        'severity': 'Medium',
        'prevention': 'Crop rotation, mulching, proper spacing',
        'treatment': 'Chlorothalonil fungicide, remove infected leaves'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'severity': 'High',
        'prevention': 'Certified seeds, hill soil, monitor weather',
        'treatment': 'Systemic fungicides, destroy infected plants'
    },
    'Tomato___healthy': {
        'name': 'Healthy Plant',
        'severity': 'None',
        'prevention': 'Continue good practices',
        'treatment': 'No treatment needed!'
    }
    # Add more diseases as needed
}

def get_disease_info(disease_key):
    """Get disease info or default"""
    if disease_key in DISEASE_INFO:
        return DISEASE_INFO[disease_key]
    return {
        'name': disease_key.replace('___', ' - ').replace('_', ' '),
        'severity': 'Unknown',
        'prevention': 'Consult agricultural expert',
        'treatment': 'Seek professional advice'
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 3
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            disease_key = class_names[idx]
            confidence = float(predictions[idx] * 100)
            info = get_disease_info(disease_key)
            
            results.append({
                'disease': info['name'],
                'confidence': round(confidence, 2),
                'severity': info['severity'],
                'prevention': info['prevention'],
                'treatment': info['treatment']
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌱 CROP DISEASE DETECTION WEB APP")
    print("="*70)
    print("\n✅ Model loaded successfully!")
    print(f"✅ Ready to detect {len(class_names)} diseases")
    print("\n🌐 Access the app at: http://localhost:5000")
    print("📱 From phone (same network): http://YOUR_IP:5000")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)