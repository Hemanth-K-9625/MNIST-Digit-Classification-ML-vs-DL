from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("models/cnn_model.h5")

@app.route("/")
def home():
    return "Digit Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert('L')

    img = image.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))

    return jsonify({"prediction": digit})

if __name__ == "__main__":
    app.run()