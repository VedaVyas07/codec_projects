"""
app.py
------
Streamlit front-end for the Handwritten Digit Recognizer.

The user can either:
    1. Draw a digit (0-9) on an in-browser canvas, or
    2. Upload a JPG/JPEG/PNG image of a handwritten digit.

The image is preprocessed with the exact same pipeline used during
training (see utils/preprocessing.py) and fed into the pretrained CNN
(model/digit_model.keras) to produce a prediction, a confidence score,
and a full probability distribution over digits 0-9.

Run with:
    streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import prepare_for_prediction

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="\U0001F522",
    layout="centered",
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "digit_model.keras")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


@st.cache_resource(show_spinner="Loading trained model ...")
def load_model():
    """Load the trained Keras model once and cache it across reruns.

    Returns None if the model file doesn't exist yet, so the caller can
    show clear setup instructions instead of crashing.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    # Imported lazily so the app can show a friendly "model missing"
    # message even in environments where importing tensorflow is slow.
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH)


def render_missing_model_instructions():
    st.error("No trained model was found.")
    st.markdown(
        """
        The application needs a trained model before it can make predictions.

        **To fix this:**
        1. Open a terminal in the project folder.
        2. Run the training script:
           ```bash
           python train_model.py
           ```
        3. Wait for training to finish (this creates `model/digit_model.keras`).
        4. Restart this app:
           ```bash
           streamlit run app.py
           ```
        """
    )


def render_prediction(display_image: np.ndarray, model_input: np.ndarray, model):
    """Run inference and render the predicted digit, confidence, probability
    bar chart, and the processed 28x28 image the model actually used.
    """
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    st.markdown("### Result")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"## Predicted Digit: **{predicted_digit}**")
        st.markdown(f"**Confidence:** {confidence:.1f}%")

        # Upscale the tiny 28x28 array with nearest-neighbor so the user
        # can clearly see the individual pixels the model received.
        preview = Image.fromarray(display_image).resize((140, 140), Image.NEAREST)
        st.image(preview, caption="Processed 28x28 image fed to the model")

    with col2:
        st.markdown("**Probability by digit**")
        prob_df = pd.DataFrame(
            {"digit": [str(d) for d in range(10)], "probability": probabilities}
        ).set_index("digit")
        st.bar_chart(prob_df)


def handle_image(pil_image: Image.Image, model):
    """Shared error-handling wrapper around preprocessing + prediction."""
    try:
        display_image, model_input = prepare_for_prediction(pil_image)
    except ValueError as exc:
        # Raised by preprocessing.py when the image has no visible strokes.
        st.warning(str(exc))
        return

    render_prediction(display_image, model_input, model)


def draw_tab(model):
    st.write("Draw a single digit (0-9) in the box below, then click **Predict**.")

    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error(
            "The `streamlit-drawable-canvas` package is required for the drawing "
            "canvas. Install it with `pip install streamlit-drawable-canvas`."
        )
        return

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    predict_clicked = st.button("Predict", type="primary", key="predict_draw")

    if predict_clicked:
        if canvas_result.image_data is None:
            st.warning("Please draw a digit before predicting.")
            return

        # image_data is an RGBA numpy array from the canvas component.
        rgba = canvas_result.image_data.astype("uint8")
        pil_image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        handle_image(pil_image, model)


def upload_tab(model):
    st.write("Upload a JPG, JPEG, or PNG image containing a single handwritten digit.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=sorted(ALLOWED_EXTENSIONS), key="digit_uploader"
    )

    if uploaded_file is not None:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
        if extension not in ALLOWED_EXTENSIONS:
            st.error(f"Unsupported file type '.{extension}'. Please upload a JPG, JPEG, or PNG image.")
            return

        try:
            pil_image = Image.open(uploaded_file)
            pil_image.load()  # force-decode now so corrupt files fail here, not later
        except UnidentifiedImageError:
            st.error("This file could not be read as an image. Please upload a valid JPG, JPEG, or PNG.")
            return
        except Exception as exc:  # noqa: BLE001 - surface any other decode error to the user
            st.error(f"Could not process this image: {exc}")
            return

        st.image(pil_image, caption="Uploaded image", width=200)

        if st.button("Predict", type="primary", key="predict_upload"):
            handle_image(pil_image, model)


def main():
    st.title("Handwritten Digit Recognizer")
    st.caption("Draw or upload a digit (0-9) and a CNN trained on MNIST will identify it.")

    model = load_model()
    if model is None:
        render_missing_model_instructions()
        st.stop()

    tab_draw, tab_upload = st.tabs(["Draw a digit", "Upload an image"])
    with tab_draw:
        draw_tab(model)
    with tab_upload:
        upload_tab(model)

    with st.expander("About this project"):
        st.write(
            "This app uses a Convolutional Neural Network (CNN) trained on the "
            "MNIST dataset of 70,000 handwritten digits. Images are converted to "
            "grayscale, resized to 28x28, normalized, and centered before being "
            "passed to the model, mirroring the preprocessing used during training."
        )


if __name__ == "__main__":
    main()
