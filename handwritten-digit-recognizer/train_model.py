"""
train_model.py
---------------
Trains a Convolutional Neural Network (CNN) to classify handwritten digits
(0-9) using the MNIST dataset, evaluates it, and saves the trained model
to disk so that app.py can load it instantly without retraining.

Usage:
    python train_model.py

Output:
    model/digit_model.keras          -- the trained Keras model
    model/confusion_matrix.png       -- confusion matrix heatmap
    model/training_history.png       -- accuracy/loss curves
"""

import gzip
import os
import sys
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")  # headless backend, no display server needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Make "utils" importable regardless of the current working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import normalize_pixels, reshape_for_cnn, IMG_SIZE

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 15
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42


# Fallback mirror used only if the standard Keras download (which pulls
# from storage.googleapis.com) is unreachable, e.g. behind a restrictive
# network/firewall. Hosts the original IDX-format MNIST files.
_MIRROR_BASE = "https://raw.githubusercontent.com/fgnt/mnist/master"
_MIRROR_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def _parse_idx_images(gz_path):
    """Parse a gzip-compressed IDX3 image file into a (N, 28, 28) uint8 array."""
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    num, rows, cols = (int.from_bytes(data[i : i + 4], "big") for i in (4, 8, 12))
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)


def _parse_idx_labels(gz_path):
    """Parse a gzip-compressed IDX1 label file into a (N,) uint8 array."""
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    num = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data[8:], dtype=np.uint8).reshape(num)


def _load_mnist_from_mirror():
    """Download (and cache) the raw MNIST IDX files from a GitHub mirror,
    used as a fallback when the primary Keras/Google-hosted download fails.
    """
    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    local_paths = {}
    for key, fname in _MIRROR_FILES.items():
        local_path = os.path.join(_DATA_CACHE_DIR, fname)
        if not os.path.exists(local_path):
            print(f"  Downloading {fname} from mirror ...")
            urllib.request.urlretrieve(f"{_MIRROR_BASE}/{fname}", local_path)
        local_paths[key] = local_path

    x_train = _parse_idx_images(local_paths["train_images"])
    y_train = _parse_idx_labels(local_paths["train_labels"])
    x_test = _parse_idx_images(local_paths["test_images"])
    y_test = _parse_idx_labels(local_paths["test_labels"])
    return (x_train, y_train), (x_test, y_test)


def load_and_preprocess_data():
    """Load MNIST and apply the shared preprocessing pipeline (normalize +
    reshape) so training-time preprocessing matches inference-time
    preprocessing exactly.
    """
    print("Loading MNIST dataset ...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as exc:  # noqa: BLE001 - network errors vary by platform
        print(f"  Standard MNIST download failed ({exc}).")
        print("  Falling back to an alternate mirror ...")
        (x_train, y_train), (x_test, y_test) = _load_mnist_from_mirror()
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Test samples:     {x_test.shape[0]}")

    # Normalize pixel values from [0, 255] integers to [0, 1] floats.
    x_train = normalize_pixels(x_train)
    x_test = normalize_pixels(x_test)

    # Reshape from (N, 28, 28) to (N, 28, 28, 1) -- CNNs expect an explicit
    # channel dimension, even for single-channel grayscale images.
    x_train = reshape_for_cnn(x_train)
    x_test = reshape_for_cnn(x_test)

    # One-hot encode integer labels (e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]) since
    # the output layer uses softmax over 10 classes with categorical
    # crossentropy loss.
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


def build_cnn_model() -> keras.Model:
    """Build a Convolutional Neural Network for 10-class digit classification.

    Architecture:
        Conv2D(32) -> ReLU -> Conv2D(32) -> ReLU -> MaxPool -> Dropout
        Conv2D(64) -> ReLU -> Conv2D(64) -> ReLU -> MaxPool -> Dropout
        Flatten -> Dense(256) -> ReLU -> Dropout -> Dense(10) -> Softmax

    Two stacked Conv2D layers per block (instead of one) let the network
    learn richer features before downsampling, which is a big part of why
    this reaches ~99% test accuracy instead of ~98%. Dropout layers reduce
    overfitting.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            # --- Convolutional block 1 ---
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # --- Convolutional block 2 ---
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # --- Classification head ---
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="digit_recognizer_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, out_path):
    """Save accuracy/loss curves so training progress can be inspected visually."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train loss")
    axes[1].plot(history.history["val_loss"], label="validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training history plot -> {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    """Compute and save a confusion matrix heatmap for the test set predictions."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Test Set")
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax)

    # Annotate each cell with its count.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {out_path}")
    return cm


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)

    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_and_preprocess_data()

    print("\nBuilding CNN model ...")
    model = build_cnn_model()
    model.summary()

    # Stop early if validation loss stops improving, and always keep the
    # best-performing weights rather than whatever the final epoch produced.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    print("\nTraining model ...")
    start = time.time()
    history = model.fit(
        x_train,
        y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f} seconds.")

    # --------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------
    print("\nEvaluating on the held-out test set ...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    plot_confusion_matrix(y_test, y_pred, os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_training_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    # --------------------------------------------------------------------
    # Save the trained model so app.py never needs to retrain.
    # --------------------------------------------------------------------
    model.save(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    # Persist a small metrics summary alongside the model for reference.
    metrics_path = os.path.join(MODEL_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Handwritten Digit Recognizer - Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Epochs run:    {len(history.history['loss'])}\n")
        f.write(f"Training time: {elapsed:.1f} seconds\n")
    print(f"Saved metrics summary -> {metrics_path}")

    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
