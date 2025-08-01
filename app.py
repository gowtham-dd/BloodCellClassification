from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from PIL import Image

# Flask config
app = Flask(__name__)
app.secret_key = 'bloodcell-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model
model = None
MODEL_INPUT_SIZE = (224, 224)  # Update based on your model's input size
try:
    model = tf.keras.models.load_model('artifacts/model_training/blood_cell_model')
    logger.info("✅ Blood Cell Classifier Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")

# Class labels
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess the blood cell image for prediction."""
    try:
        image = Image.open(image_path).convert("RGB").resize(MODEL_INPUT_SIZE)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(image))
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            flash('Model not available. Please contact support.', 'error')
            return redirect(url_for('predict_form'))

        if 'bloodcell' not in request.files:
            flash('No blood cell image uploaded.', 'error')
            return redirect(url_for('predict_form'))

        file = request.files['bloodcell']
        if file.filename == '':
            flash('No selected file.', 'error')
            return redirect(url_for('predict_form'))

        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            processed_image = preprocess_image(filepath)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)

            result = {
                'class': CLASS_NAMES[predicted_class],
                'confidence': round(confidence, 2),
                'image_path': filepath,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'all_predictions': {
                    CLASS_NAMES[i]: round(float(predictions[0][i] * 100), 2)
                    for i in range(len(CLASS_NAMES))
                }
            }

            return render_template('result.html', result=result)

        else:
            flash('Allowed file types: png, jpg, jpeg, bmp', 'error')
            return redirect(url_for('predict_form'))

    except Exception as e:
        logger.error(f'Prediction error: {e}')
        flash('Something went wrong during prediction. Try again.', 'error')
        return redirect(url_for('predict_form'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if not model:
            return jsonify({'success': False, 'error': 'Model not available'}), 500

        if 'bloodcell' not in request.files:
            return jsonify({'success': False, 'error': 'No blood cell image uploaded'}), 400

        file = request.files['bloodcell']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            processed_image = preprocess_image(filepath)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)

            # Optionally remove the file after prediction
            os.remove(filepath)

            return jsonify({
                'success': True,
                'prediction': CLASS_NAMES[predicted_class],
                'confidence': round(confidence, 2),
                'timestamp': datetime.now().isoformat(),
                'all_predictions': {
                    CLASS_NAMES[i]: round(float(predictions[0][i] * 100), 2)
                    for i in range(len(CLASS_NAMES))
                }
            })

        else:
            return jsonify({'success': False, 'error': 'Allowed file types: png, jpg, jpeg, bmp'}), 400

    except Exception as e:
        logger.error(f"API Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Prediction failed due to internal error'}), 500

@app.route('/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': bool(model)
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Run Server
if __name__ == "__main__":
    print("🧫 Starting Blood Cell Classification App")
    print(f"🔧 Model: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print("🚀 Running at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)