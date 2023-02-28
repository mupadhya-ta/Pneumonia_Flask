from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow.lite as tflite
import os

app = Flask(__name__)

# Load the TFLite model
model_path = "models/pneumonia.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    img = tflite.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    img = tflite.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.float32(img)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run the model
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)

    # Get the predicted label
    predicted_label = np.argmax(output_data)
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    predicted_class = class_names[predicted_label]

    # Return the prediction result as JSON
    return jsonify({
        'predicted_class': predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)
