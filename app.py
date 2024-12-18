import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'model_after_testing.keras'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load pre-trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    try:
        # Load image
        img = load_img(file_path, target_size=(224, 224))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Machine learning model not loaded. Please contact support.'
            }), 500

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file type
        if file and allowed_file(file.filename):
            # Save file with Unicode support
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure Unicode filename support
            filepath = filepath.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Ensure directory exists with Unicode support
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save file
            file.save(filepath)
            
            # Preprocess image
            processed_img = preprocess_image(filepath)
            
            if processed_img is None:
                return jsonify({'error': 'Failed to process image'}), 400
            
            # Make prediction
            prediction = model.predict(processed_img)[0][0]
            
            # Determine result
            result = "Eye Flu Detected" if prediction >= 0.5 else "Healthy Eye"
            
            # Remove uploaded file
            try:
                os.remove(filepath)
            except Exception as remove_error:
                print(f"Error removing file: {remove_error}")
            
            return jsonify({
                'result': result, 
                'confidence': float(prediction)
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Set the default encoding to UTF-8
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # For Python 3.7+
    sys.stderr.reconfigure(encoding='utf-8')  # For Python 3.7+
    
    app.run(debug=True)