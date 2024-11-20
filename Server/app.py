from flask import Flask, request, jsonify
import io
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras

app = Flask(__name__)

# Define constants
MODEL_PATH = './gtsrb_model.h5'
IMG_WIDTH, IMG_HEIGHT = 64, 64

# Load the model
model = keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']  # Get the FileStorage object

    try:
        # Convert the FileStorage object to a BytesIO object
        image_bytes = io.BytesIO(image_file.read())  # Read file data as bytes
        img = load_img(image_bytes, target_size=(IMG_WIDTH, IMG_HEIGHT))  # Use BytesIO for load_img
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Expand dimensions for model input shape

        # Predict the class
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
