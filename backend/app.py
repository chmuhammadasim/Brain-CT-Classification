from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/final_model.h5')  # Or .keras

CLASS_NAMES = ['aneurysm', 'cancer', 'tumor']  # Replace with actual class names

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(img_path):
    """Preprocess the image for model prediction."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, bmp, gif'}), 400

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    try:
        img = preprocess(path)
        prediction = model.predict(img)[0]
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        probabilities = {CLASS_NAMES[i]: float(f"{prob*100:.2f}") for i, prob in enumerate(prediction)}
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({
        'prediction': CLASS_NAMES[index],
        'confidence': f'{confidence * 100:.2f}%',
        'probabilities': probabilities,
        'file_path': path
    })

if __name__ == '__main__':
    app.run(debug=True)
