import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model safely
model = load_model("models/cnn_model.h5", compile=False)

st.title("🔢 Handwritten Digit Classifier")

st.write("Upload a digit image (28x28 or similar)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess
    img = image.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
