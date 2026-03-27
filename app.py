from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# FIX 1: load model safely
model = load_model("models/cnn_model.h5", compile=False)

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

# FIX 2: bind to Render port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
