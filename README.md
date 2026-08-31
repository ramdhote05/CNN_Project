# Keyboard vs Mouse Image Classification

This project is a simple image-classification demo that predicts whether an uploaded image looks more like a keyboard or a mouse.

## Features

- Upload an image through a Streamlit web interface
- Preprocess the uploaded image
- Extract image features and classify it into a binary label
- Show the predicted class and confidence score
- Run in a lightweight, cloud-friendly setup without TensorFlow

## Current Model Behavior

The app currently classifies between two classes:

- Keyboard
- Mouse

This is a lightweight demonstration project and is intended for learning and experimentation.

## Limitations

This project has important limitations that users should know:

- It is not a general-purpose object detector
- It is not trained to recognize ID cards, documents, or unrelated objects
- It may give incorrect results on images with different backgrounds, lighting, resizing, or angles
- It is trained on a limited dataset and should not be treated as a production-grade classifier
- Predictions depend heavily on image similarity to the training examples

In other words, if an uploaded image is not close to the keyboard/mouse training distribution, the model may still produce a forced guess rather than a meaningful result.

## Repository Structure

- [app.py](app.py) — Streamlit web app for image upload and prediction
- [keyboard_vs_mouse_image_classification.py](keyboard_vs_mouse_image_classification.py) — training pipeline and experiment code
- [Keyboard_dataset](Keyboard_dataset) — keyboard sample images
- [Mouse_dataset](Mouse_dataset) — mouse sample images
- [combined_dataset](combined_dataset) — combined dataset used for model experiments
- [requirements.txt](requirements.txt) — project dependencies
- [runtime.txt](runtime.txt) — runtime hint for deployment environments
- [test_image.jpg](test_image.jpg) — sample test image

## Setup

1. Clone the repository
2. Create a virtual environment if needed
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Deployment Notes

This app is designed to be simple and compatible with deployment environments that do not support TensorFlow on newer Python versions. The current version uses a lightweight image feature classifier for better cloud compatibility.

## Notes

This project is best viewed as a beginner-friendly machine learning demo. It can be expanded with more classes, a larger dataset, and a stronger CNN model for more accurate predictions.

## License

This project is intended for educational and experimental use.
