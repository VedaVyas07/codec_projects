"""
Predicts the fruit class for a single image via the command line.

Usage:
    python src/predict.py --image path/to/image.jpg
"""

import os
import sys
import argparse
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CLASS_NAMES, MODEL_PATH, preprocess_image


def predict_image(image_path, model=None):
    """Returns (predicted_class, confidence, all_class_probabilities)."""
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run src/train.py first."
            )
        model = load_model(MODEL_PATH)

    processed = preprocess_image(image_path)
    probabilities = model.predict(processed, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    class_probs = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }
    return predicted_class, confidence, class_probs


def main():
    parser = argparse.ArgumentParser(description="Predict fruit class from an image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    predicted_class, confidence, class_probs = predict_image(args.image)

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%\n")
    print("All class probabilities:")
    for cls, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:10s}: {prob * 100:5.2f}%")


if __name__ == "__main__":
    main()