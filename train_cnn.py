from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "combined_dataset"
MODEL_PATH = BASE_DIR / "keyboard_mouse_cnn.keras"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    model = build_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)
    model.save(MODEL_PATH)

    print("Training complete.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Final val accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")
    train_model()
