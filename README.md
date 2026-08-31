# Keyboard vs Mouse Image Classification

This project uses a Convolutional Neural Network (CNN) to classify images as either a keyboard or a mouse.

## Project Overview

The model is trained on a custom dataset of keyboard and mouse images and predicts the class of an input image.

## Repository Structure

- `keyboard_vs_mouse_image_classification.py` — training and evaluation pipeline for the CNN model
- `keyboard_mouse_cnn.keras` — trained Keras model
- `combined_dataset/` — organized dataset used for model training
- `Keyboard_dataset/` — original keyboard images
- `Mouse_dataset/` — original mouse images
- `test_image.jpg` — sample image for testing the model
- `requirements.txt` — required Python packages

## Setup

1. Clone the repository
2. Create a virtual environment (optional but recommended)
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

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

## Usage

Run the notebook/script:

```bash
python keyboard_vs_mouse_image_classification.py
```

This script:
- loads the dataset,
- prepares the image data,
- trains the CNN,
- validates the model,
- and evaluates overall accuracy.

## Model Details

- Input size: 128 x 128 RGB images
- Model type: CNN built with TensorFlow/Keras
- Classes: Keyboard, Mouse
- Output: binary image classification

## Dataset

The dataset is divided into two folders:

- `Keyboard_dataset`
- `Mouse_dataset`

These images are combined into a single structure under `combined_dataset` for training.

## Notes

This project was created for image classification experimentation and can be extended with additional classes, data augmentation, or transfer learning.

## License

This project is for educational and experimental use.
