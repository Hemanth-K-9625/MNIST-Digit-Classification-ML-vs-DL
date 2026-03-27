import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model
model = joblib.load("models/logistic_model.pkl")

st.title("🔢 Digit Classifier")

st.write("Using Logistic Regression (lightweight deployment)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Input Image", width=150)

    # Preprocess
    img = image.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    st.success(f"Predicted Digit: {prediction[0]}")
