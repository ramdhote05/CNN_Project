# Keyboard vs Mouse Image Classification

This project uses a Convolutional Neural Network (CNN) to classify images as either a keyboard or a mouse.

## Project Overview

The model is trained on a custom dataset of keyboard and mouse images and predicts the class of an input image.

## Repository Structure

- [app.py](app.py) — Streamlit web app for uploading and classifying images
- [keyboard_vs_mouse_image_classification.py](keyboard_vs_mouse_image_classification.py) — model training and evaluation pipeline
- [keyboard_mouse_cnn.keras](keyboard_mouse_cnn.keras) — trained Keras model
- [combined_dataset](combined_dataset) — organized dataset used for model training
- [Keyboard_dataset](Keyboard_dataset) — original keyboard images
- [Mouse_dataset](Mouse_dataset) — original mouse images
- [test_image.jpg](test_image.jpg) — sample image for testing the model
- [requirements.txt](requirements.txt) — required Python packages

## Setup

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

```bash
streamlit run app.py
```

The app lets you upload an image and predicts whether it contains a keyboard or a mouse.

## Dependencies

The project requires libraries such as:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Streamlit

## Model Details

- Input size: 128 x 128 RGB images
- Model type: CNN built with TensorFlow/Keras
- Classes: Keyboard, Mouse
- Output: binary image classification

## Dataset

The dataset is divided into two folders:

- [Keyboard_dataset](Keyboard_dataset)
- [Mouse_dataset](Mouse_dataset)

These images are combined into a single structure under [combined_dataset](combined_dataset) for training.

## Notes

This project was created for image classification experimentation and can be extended with additional classes, data augmentation, or transfer learning.

## License

This project is for educational and experimental use.
