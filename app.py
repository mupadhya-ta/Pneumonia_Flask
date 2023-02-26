from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
from keras.models import load_model
import numpy as np


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/pneumonia.h5")

@app.route('/', methods=['GET'])
def home():
    return "Hello From Flask"
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    
    # Save the file to disk
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Load the image and preprocess it
    img = tf.keras.utils.load_img(file_path, target_size=(128,128))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    # Make a prediction with the model
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    class_names={0: 'NORMAL', 1: 'PNEUMONIA'}
    predicted_class = class_names[predicted_label]

    # Return the prediction result as JSON
    return jsonify({
        'predicted_class': predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)
