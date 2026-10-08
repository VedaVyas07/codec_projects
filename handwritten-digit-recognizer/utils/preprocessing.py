"""
preprocessing.py
-----------------
Shared image preprocessing utilities used by BOTH the training pipeline
(train_model.py) and the inference pipeline (app.py). Keeping this logic
in one place guarantees that images fed to the model at prediction time
are transformed exactly the same way the MNIST training images were,
which is essential for the model to perform well on real user input.

MNIST convention:
    - Images are 28x28 grayscale.
    - The digit (foreground/stroke) is WHITE (high pixel values, ~255).
    - The background is BLACK (low pixel values, ~0).
    - Pixel values are normalized to the [0, 1] float range before being
      fed into the CNN.
"""

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 28


def normalize_pixels(image_array: np.ndarray) -> np.ndarray:
    """Scale pixel values from the 0-255 range down to 0-1.

    Both training (raw MNIST arrays) and inference (uploaded/drawn images)
    pass through this exact same function so the model always sees inputs
    on the same numeric scale it was trained on.
    """
    return image_array.astype("float32") / 255.0


def reshape_for_cnn(image_array: np.ndarray) -> np.ndarray:
    """Reshape a batch of 28x28 grayscale images into (N, 28, 28, 1),
    the 4D tensor shape Conv2D layers expect (batch, height, width, channels).
    """
    return image_array.reshape((-1, IMG_SIZE, IMG_SIZE, 1))


def _autoinvert_if_needed(gray_image: Image.Image) -> Image.Image:
    """
    MNIST digits are WHITE strokes on a BLACK background. Most real-world
    input is the opposite: dark ink/pencil strokes on a light/white
    background (a photographed page, or a white drawing canvas with a
    dark pen color). We inspect the average brightness of the image to
    guess which case we're in and invert colors when the background looks
    light, so the result always matches the MNIST convention before it
    reaches the model.
    """
    pixels = np.array(gray_image)
    if pixels.mean() > 127:  # background is bright -> ink-on-paper style
        gray_image = ImageOps.invert(gray_image)
    return gray_image


def _center_by_mass(digit_28x28: np.ndarray) -> np.ndarray:
    """
    Re-center the digit inside the 28x28 frame using its center of mass,
    mirroring how the original MNIST dataset itself was constructed. This
    makes off-center drawings/uploads much easier for the CNN to classify,
    since the network was trained exclusively on centered digits.
    """
    total = digit_28x28.sum()
    if total == 0:
        return digit_28x28

    rows, cols = np.indices(digit_28x28.shape)
    row_com = (rows * digit_28x28).sum() / total
    col_com = (cols * digit_28x28).sum() / total

    row_shift = int(round(digit_28x28.shape[0] / 2.0 - row_com))
    col_shift = int(round(digit_28x28.shape[1] / 2.0 - col_com))

    centered = np.roll(digit_28x28, row_shift, axis=0)
    centered = np.roll(centered, col_shift, axis=1)
    return centered


def preprocess_pil_image(pil_image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline for an arbitrary PIL image (a canvas
    drawing or an uploaded photo) so that it matches the MNIST training
    distribution as closely as possible.

    Steps:
        1. Convert to grayscale.
        2. Invert colors if the background is light (see _autoinvert_if_needed).
        3. Crop tightly to the digit's bounding box (removes empty margins).
        4. Resize the digit to fit inside a 20x20 box, preserving aspect
           ratio, then paste onto a blank 28x28 canvas (this is the same
           construction procedure used to build the original MNIST images).
        5. Re-center using center of mass.

    Returns:
        A (28, 28) uint8 numpy array, ready for normalize_pixels() +
        reshape_for_cnn().

    Raises:
        ValueError: if the image appears to be blank (an "empty drawing"
            or a photo with no visible strokes).
    """
    gray = pil_image.convert("L")
    gray = _autoinvert_if_needed(gray)
    arr = np.array(gray)

    # Threshold to find the digit's bounding box. Anything reasonably
    # brighter than background-black is considered part of the stroke.
    threshold = 20
    mask = arr > threshold
    if not mask.any():
        raise ValueError("The image appears to be blank. Please draw or upload a digit.")

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    digit = arr[row_min : row_max + 1, col_min : col_max + 1]

    # Resize so the longest side becomes 20px, preserving aspect ratio,
    # then center it on a black 28x28 canvas (leaves a small margin, just
    # like real MNIST digits do).
    digit_img = Image.fromarray(digit)
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, round(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, round(h * (20 / w)))
    digit_img = digit_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    upper_left = ((IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2)
    canvas.paste(digit_img, upper_left)

    final_arr = np.array(canvas)
    final_arr = _center_by_mass(final_arr)

    return final_arr.astype("uint8")


def prepare_for_prediction(pil_image: Image.Image):
    """
    Convenience wrapper used by app.py: runs the full preprocessing
    pipeline and returns both:
        - display_image: the (28, 28) uint8 array (used to show the user
          "what the model actually sees")
        - model_input: the (1, 28, 28, 1) normalized float32 array, ready
          to be passed straight into model.predict()
    """
    display_image = preprocess_pil_image(pil_image)
    normalized = normalize_pixels(display_image)
    model_input = reshape_for_cnn(normalized)
    return display_image, model_input
