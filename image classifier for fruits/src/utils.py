"""
Shared constants and helper functions used across
train.py, evaluate.py, predict.py, and app.py
"""

import os
import json
import numpy as np
from PIL import Image

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 32

CLASS_NAMES = [
    "apple",
    "banana",
    "orange",
    "mango",
    "grapes"
]

# ----------------------------
# Project Paths
# ----------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "train"
)

TEST_DIR = os.path.join(
    BASE_DIR, "data", "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "fruit_classifier_model.h5"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

CLASS_INDICES_PATH = os.path.join(
    OUTPUT_DIR, "class_indices.json"
)


# ----------------------------
# Create Required Directories
# ----------------------------
def ensure_dirs():
    """Create model and output directories if they don't exist."""
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image, img_size=IMG_SIZE):
    """
    Preprocess a single image for prediction.

    Accepts:
    - PIL.Image
    - File path (str)
    - NumPy array

    Returns:
    - Normalized NumPy array of shape:
      (1, img_size, img_size, 3)
    """

    # ----------------------------
    # If image is a file path
    # ----------------------------
    if isinstance(image, str):

        try:
            img = Image.open(image).convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Could not read image at path: {image}"
            ) from e

    # ----------------------------
    # If image is already a PIL image
    # ----------------------------
    elif isinstance(image, Image.Image):

        img = image.convert("RGB")

    # ----------------------------
    # If image is a NumPy array
    # ----------------------------
    elif isinstance(image, np.ndarray):

        img = image

        # RGBA → RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        # Convert NumPy array to PIL image
        img = Image.fromarray(img.astype("uint8")).convert("RGB")

    # ----------------------------
    # Unsupported image type
    # ----------------------------
    else:

        raise TypeError(
            "Unsupported image type for preprocessing."
        )

    # ----------------------------
    # Resize image using PIL
    # ----------------------------
    img = img.resize(
        (img_size, img_size),
        Image.Resampling.LANCZOS
    )

    # ----------------------------
    # Convert image to NumPy array
    # ----------------------------
    img = np.array(
        img,
        dtype=np.float32
    )

    # ----------------------------
    # Normalize pixel values
    # ----------------------------
    img = img / 255.0

    # ----------------------------
    # Add batch dimension
    # ----------------------------
    img = np.expand_dims(
        img,
        axis=0
    )

    return img