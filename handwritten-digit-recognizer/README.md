# Handwritten Digit Recognizer

A complete, runnable machine learning application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset, served through an interactive Streamlit web app.

## 1. Project Overview

This project trains a CNN on the classic MNIST dataset of handwritten digits and wraps it in a Streamlit web application. Users can either draw a digit on an in-browser canvas or upload a photo of a handwritten digit, and the app returns the model's prediction along with a confidence score and a full probability breakdown across all 10 digit classes.

The project is split into two independent stages: a training script (`train_model.py`) that produces a saved model file, and an inference app (`app.py`) that loads that saved model and serves predictions. This mirrors how real ML systems are structured in practice — training and serving are decoupled, so the app starts instantly instead of retraining on every launch.

## 2. Problem Statement

Recognizing handwritten digits is a foundational computer vision problem: given a small grayscale image of a digit, correctly classify it as one of 0-9. It's a well-understood, well-benchmarked task that makes an ideal starting point for learning CNNs, image preprocessing, and end-to-end ML application design, while still requiring careful handling of real-world messiness — off-center strokes, inverted colors, varying image formats, and blank/invalid input.

## 3. Features

- CNN trained from scratch on MNIST, reaching **99.23% test accuracy**.
- Model is trained once and saved to disk (`model/digit_model.keras`); the app loads it instantly rather than retraining on every launch.
- Two input methods in the Streamlit UI:
  - **Draw a digit** on an interactive canvas.
  - **Upload an image** (JPG, JPEG, PNG) of a handwritten digit.
- Robust preprocessing pipeline shared identically between training and inference: grayscale conversion, automatic background/foreground color inversion, bounding-box cropping, aspect-ratio-preserving resize, and center-of-mass recentering — so the input the model sees always matches the MNIST convention it was trained on.
- Prediction output includes the predicted digit, a confidence percentage, a bar chart of probabilities across all 10 digits, and a view of the exact 28x28 processed image fed into the model.
- Graceful error handling for a missing trained model, invalid/corrupt image files, unsupported formats, and blank drawings/uploads.
- Evaluation artifacts (confusion matrix and training curves) are generated automatically during training.

## 4. Technologies Used

- Python 3.11+
- TensorFlow / Keras — CNN model definition, training, and inference
- NumPy — array/image manipulation
- Matplotlib — training curves and confusion matrix visualization
- Pillow (PIL) — image loading, format conversion, resizing
- Streamlit — web application frontend
- streamlit-drawable-canvas — the in-browser drawing canvas component
- scikit-learn — confusion matrix and classification report metrics
- pandas — structuring probability data for the Streamlit bar chart

## 5. Dataset Information

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels:

- **60,000 training images**
- **10,000 test images**

Each image is a centered, size-normalized digit on a black background with a white stroke. `train_model.py` loads it via `keras.datasets.mnist.load_data()` (with an automatic fallback to a mirrored copy of the raw IDX files if that primary download is ever blocked by a restrictive network, e.g. a locked-down corporate firewall).

## 6. CNN Architecture

```
Input (28x28x1 grayscale)
 -> Conv2D(32, 3x3, ReLU, same padding)
 -> Conv2D(32, 3x3, ReLU, same padding)
 -> MaxPooling2D(2x2)
 -> Dropout(0.25)
 -> Conv2D(64, 3x3, ReLU, same padding)
 -> Conv2D(64, 3x3, ReLU, same padding)
 -> MaxPooling2D(2x2)
 -> Dropout(0.25)
 -> Flatten
 -> Dense(256, ReLU)
 -> Dropout(0.5)
 -> Dense(10, Softmax)
```

- **Total parameters:** 870,634
- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Callbacks:** `EarlyStopping` (stops training once validation loss stops improving and restores the best-performing weights) and `ReduceLROnPlateau` (halves the learning rate when progress stalls)
- Two stacked `Conv2D` layers per block let the network learn richer local features before each downsampling step, which noticeably improves accuracy over a single-conv-per-block design.

## 7. Project Structure

```
handwritten-digit-recognizer/
├── app.py                   # Streamlit web application (inference)
├── train_model.py           # Training script (run once to produce the model)
├── model/
│   ├── digit_model.keras    # Saved trained model (created by train_model.py)
│   ├── confusion_matrix.png # Confusion matrix on the test set
│   ├── training_history.png # Accuracy/loss curves
│   └── metrics.txt          # Plain-text summary of final test metrics
├── utils/
│   ├── __init__.py
│   └── preprocessing.py     # Preprocessing shared by training AND inference
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd handwritten-digit-recognizer
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 9. How to Train the Model

```bash
python train_model.py
```

This will:

1. Download the MNIST dataset (cached locally after the first run).
2. Preprocess it (normalize to [0, 1], reshape to `(N, 28, 28, 1)`).
3. Build and train the CNN described above.
4. Evaluate it on the 10,000-image test set and print test accuracy, test loss, and a full classification report.
5. Save `model/confusion_matrix.png` and `model/training_history.png` for visual inspection.
6. Save the trained model to `model/digit_model.keras`, and a metrics summary to `model/metrics.txt`.

A pretrained model is already included in `model/digit_model.keras`, so this step is optional unless you want to retrain from scratch or experiment with the architecture.

## 10. How to Run the Application

```bash
streamlit run app.py
```

This opens the app in your browser (typically at `http://localhost:8501`). If `model/digit_model.keras` doesn't exist yet, the app will detect this and display clear on-screen instructions to run `train_model.py` first, instead of crashing.

## 11. How Prediction Works

1. The user either draws a digit on the canvas or uploads a JPG/JPEG/PNG image.
2. The image is converted to grayscale.
3. The app automatically detects whether the image looks like "ink on paper" (dark strokes on a light background) and inverts it if so, so it always matches MNIST's white-stroke-on-black-background convention.
4. The digit is cropped to its bounding box, resized to fit within a 20x20 box (preserving aspect ratio), and pasted onto a blank 28x28 canvas — the same construction procedure used to build the original MNIST dataset.
5. The digit is re-centered using its center of mass.
6. Pixel values are normalized from [0, 255] to [0, 1] and reshaped to `(1, 28, 28, 1)`.
7. This tensor is fed into the trained CNN, which outputs a probability for each of the 10 digit classes via softmax.
8. The app displays the highest-probability digit, its confidence percentage, a bar chart of all 10 probabilities, and the exact 28x28 image the model used — so you can sanity-check what the network actually "saw."

Because `utils/preprocessing.py` is imported by both `train_model.py` and `app.py`, the exact same transformation logic is guaranteed to apply at both training and prediction time.

## 12. Expected Results

On the held-out MNIST test set (10,000 images), this project's included model achieves:

- **Test Accuracy:** 99.23%
- **Test Loss:** 0.0228

Per-digit precision/recall are all above 97%, with most digits above 99%. See `model/confusion_matrix.png` for the full breakdown after training.

Real-world drawn/uploaded digits will typically see somewhat lower accuracy than the clean MNIST test set, since handwriting style, image noise, and camera artifacts vary far more than MNIST's curated samples — this is expected and is why the preprocessing pipeline (auto-inversion, cropping, and recentering) exists to close that gap as much as possible.

## 13. Future Improvements

- Data augmentation (small rotations, shifts, zooms) during training to further improve robustness to real-world handwriting variation.
- Support for recognizing multi-digit strings rather than a single digit at a time.
- Model architecture experiments (batch normalization, residual connections, or a small ensemble) to push accuracy further.
- Deploy the app publicly (Streamlit Community Cloud, Docker, etc.) instead of running locally only.
- Add a feedback mechanism where users confirm/correct predictions, enabling active learning from real usage data.
- Export a TensorFlow Lite version for lightweight mobile/edge deployment.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| App shows "No trained model was found" | `model/digit_model.keras` is missing | Run `python train_model.py` |
| "This file could not be read as an image" | Uploaded file is corrupted or not a real image | Upload a valid JPG/JPEG/PNG |
| "Unsupported file type" | File extension isn't jpg/jpeg/png | Convert or re-export the image to a supported format |
| "The image appears to be blank" | Nothing was drawn, or the uploaded photo has no visible dark strokes | Draw a clearer digit or upload a higher-contrast photo |
| MNIST download fails during training | Network/firewall blocks the default host | `train_model.py` automatically retries via a mirrored dataset source |
