import os

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('models/pneumonia.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Save the file to disk
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Load the image and preprocess it
    img = Image.open(file_path)
    img = img.convert('RGB')  # Convert image to RGB format
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0

    # Make a prediction
    predicted_prob = model.predict(np.expand_dims(img_array, axis=0))[0]
    predicted_label = np.argmax(predicted_prob)
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    predicted_class = class_names[predicted_label]

    # Return the prediction result as JSON
    return jsonify({
        'predicted_class': predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)

    