import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keyboard_mouse_model.pkl"
DATASET_FOLDERS = {
    "Keyboard": BASE_DIR / "Keyboard_dataset",
    "Mouse": BASE_DIR / "Mouse_dataset",
}
TARGET_SIZE = (32, 32)


def image_to_features(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE)
    gray = np.asarray(image, dtype=np.float32).mean(axis=2) / 255.0
    return gray.reshape(-1)


def list_image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )


def train_model():
    X, y = [], []

    for label, folder in DATASET_FOLDERS.items():
        for image_path in list_image_files(folder):
            img = Image.open(image_path)
            X.append(image_to_features(img))
            y.append(label)

    if not X:
        raise FileNotFoundError("No training images were found in Keyboard_dataset or Mouse_dataset.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(np.vstack(X), y)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    return model


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)
    return train_model()


def predict_class(model, image: Image.Image):
    features = image_to_features(image)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = float(np.max(probabilities))
    return prediction, confidence


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

st.caption("Model source: keyboard_mouse_model.pkl")
