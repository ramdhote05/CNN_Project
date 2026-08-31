from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keyboard_mouse_cnn.keras"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Keyboard", "Mouse"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_class(model, image: Image.Image):
    processed = preprocess_image(image)
    probability = model.predict(processed, verbose=0)[0]
    predicted_index = int(np.argmax(probability))
    label = CLASS_NAMES[predicted_index]
    confidence = float(probability[predicted_index])
    return label, confidence


st.set_page_config(page_title="Keyboard vs Mouse Classifier", layout="wide")
st.title("Keyboard vs Mouse Image Classifier")
st.write("Upload an image and the model will predict whether it shows a keyboard or a mouse.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        model = load_model()
        label, confidence = predict_class(model, image)
        st.subheader("Prediction")
        st.metric("Class", label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.success(f"The model predicts this is a {label.lower()}.")
else:
    st.info("Please upload an image to get a prediction.")

st.caption("Model source: keyboard_mouse_cnn.keras")
