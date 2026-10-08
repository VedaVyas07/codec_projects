"""
Fruit Image Classifier - Streamlit UI

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from src.utils import CLASS_NAMES, MODEL_PATH, preprocess_image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    /* How it works boxes */
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        min-height: 145px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.06);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return load_model(MODEL_PATH)


# =========================================================
# FRUIT ICONS
# =========================================================

fruit_icons = {
    "apple": "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "mango": "🥭",
    "grapes": "🍇"
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🍎 Fruit Classifier")

    st.markdown("---")

    # About
    st.markdown("### 📌 About")

    st.write(
        "This application uses a deep learning image "
        "classification model to identify fruits from "
        "uploaded images."
    )

    # Supported fruits
    st.markdown("### 🍓 Supported Fruits")

    for fruit in CLASS_NAMES:

        icon = fruit_icons.get(
            fruit,
            "🍓"
        )

        st.write(
            f"{icon} **{fruit.capitalize()}**"
        )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🍎 Fruit Image Classifier'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a fruit image and let the AI model identify it'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = get_model()

if model is None:

    st.error(
        "⚠️ Trained model not found. "
        "Please run the training script first."
    )

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📸 Upload Fruit Image'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    key="fruit_image_uploader"
)


# =========================================================
# IMAGE + ANALYSIS
# =========================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Create two columns
    col1, col2 = st.columns(2)


    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    with col1:

        st.markdown(
            "### 🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # =====================================================
    # ANALYSIS
    # =====================================================

    with col2:

        st.markdown(
            "### 🔍 Analysis"
        )

        st.write(
            "The AI model will analyze the uploaded image "
            "and predict the fruit."
        )

        # Classify button
        classify = st.button(
            "🔮 Classify Fruit",
            key="classify_button"
        )


        # =================================================
        # PREDICTION
        # =================================================

        if classify:

            with st.spinner(
                "Analyzing image..."
            ):

                # Preprocess image
                processed = preprocess_image(
                    image
                )

                # Model prediction
                probabilities = model.predict(
                    processed,
                    verbose=0
                )[0]

                # Find highest probability
                predicted_idx = int(
                    np.argmax(probabilities)
                )

                # Get predicted fruit
                predicted_class = CLASS_NAMES[
                    predicted_idx
                ]

                # Get confidence
                confidence = float(
                    probabilities[
                        predicted_idx
                    ]
                )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            icon = fruit_icons.get(
                predicted_class,
                "🍓"
            )

            st.success(
                f"{icon} Prediction: "
                f"{predicted_class.capitalize()}"
            )

            # Confidence value
            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

            # Confidence progress bar
            st.progress(
                confidence,
                text=(
                    f"Model confidence: "
                    f"{confidence * 100:.2f}%"
                )
            )


            # =================================================
            # CONFIDENCE SCORES
            # =================================================

            st.markdown(
                "### 📊 Confidence Scores"
            )

            # Create dataframe
            df = pd.DataFrame({

                "Fruit": [
                    c.capitalize()
                    for c in CLASS_NAMES
                ],

                "Confidence (%)": [
                    probabilities[i] * 100
                    for i in range(
                        len(CLASS_NAMES)
                    )
                ]

            })


            # Sort highest confidence first
            df = df.sort_values(
                "Confidence (%)",
                ascending=False
            )


            # =================================================
            # BAR CHART
            # =================================================

            chart_df = df.set_index(
                "Fruit"
            )

            st.bar_chart(
                chart_df
            )


            # =================================================
            # CONFIDENCE TABLE
            # =================================================

            st.dataframe(
                df.style.format({
                    "Confidence (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("---")

st.markdown(
    "### 🚀 How It Works"
)

col1, col2, col3 = st.columns(3)


# =========================================================
# STEP 1
# =========================================================

with col1:

    st.markdown(
        """
        <div class="info-box">

        <h4>📤 1. Upload</h4>

        <p>
        Upload an image of an apple, banana, orange,
        mango, or grapes.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STEP 2
# =========================================================

with col2:

    st.markdown(
        """
        <div class="info-box">

        <h4>🧠 2. Analyze</h4>

        <p>
        The MobileNetV2 deep learning model
        analyzes the uploaded image.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STEP 3
# =========================================================

with col3:

    st.markdown(
        """
        <div class="info-box">

        <h4>🎯 3. Predict</h4>

        <p>
        The model predicts the fruit and provides
        a confidence score.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("---")
st.markdown("### 💡 Project Information")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("🍎 Fruit Classes", "5")

with info2:
    st.metric("🧠 Model", "MobileNetV2")

with info3:
    st.metric("🎯 Test Accuracy", "78%")

with info4:
    st.metric("⚡ Framework", "TensorFlow")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🍎 Fruit Image Classifier
    <br>
    Deep Learning & Transfer Learning Project

    </div>
    """,
    unsafe_allow_html=True
)