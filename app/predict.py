"""
app/predict.py
──────────────
Prediction module for the Cancer Detection Streamlit app.

Functions
---------
load_model          – cached model loader
preprocess_image    – PIL Image → (1,224,224,3) float32 array
predict             – run inference, return structured dict
categorize_confidence – map probability to High/Medium/Low label
validate_image      – validate & open an UploadedFile as PIL.Image
"""

import io
import numpy as np
import streamlit as st
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model loader (cached — loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load a Keras .h5 model and cache it so it is only loaded once per
    Streamlit session, regardless of how many re-runs occur.

    Args:
        model_path: Absolute or relative path to the .h5 model file.

    Returns:
        tf.keras.Model
    """
    import tensorflow as tf  # import here to keep startup fast when unused

    model = tf.keras.models.load_model(model_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image pre-processing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize a PIL Image to 224×224, convert to float32 in [0, 1], and add
    a batch dimension.

    Args:
        image: PIL.Image (any mode)

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, img_array: np.ndarray) -> dict:
    """
    Run binary cancer detection inference on a pre-processed image array.

    Decision logic
    --------------
    THRESHOLD = 0.65  (raised from 0.5 to reduce false positives)

    prob > 0.75  → "Model Assessment: Likely Cancer"    (High confidence)
    prob > 0.55  → "Model Assessment: Uncertain"         (Medium — borderline)
    otherwise    → "Model Assessment: Likely Non-Cancer" (Low cancer probability)

    Args:
        model:     Loaded tf.keras.Model (sigmoid output head)
        img_array: np.ndarray shape (1, 224, 224, 3)

    Returns:
        dict with keys:
            probability          – float in [0, 1], cancer probability
            non_cancer_prob      – float in [0, 1], i.e. 1 - probability
            label                – human-readable assessment string
            confidence_pct       – cancer probability as percentage
            non_cancer_pct       – non-cancer probability as percentage
            confidence_level     – "High" | "Medium" | "Low"
    """
    THRESHOLD = 0.65

    raw = model.predict(img_array, verbose=0)
    probability = float(raw[0][0])
    non_cancer_prob = 1.0 - probability

    if probability > 0.75:
        label = "Model Assessment: Likely Cancer"
        level = "High"
    elif probability > 0.55:
        label = "Model Assessment: Uncertain"
        level = "Medium"
    else:
        label = "Model Assessment: Likely Non-Cancer"
        level = "Low"

    return {
        "probability":      probability,
        "non_cancer_prob":  non_cancer_prob,
        "label":            label,
        "confidence_pct":   round(probability * 100, 1),
        "non_cancer_pct":   round(non_cancer_prob * 100, 1),
        "confidence_level": level,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Confidence categorisation
# ─────────────────────────────────────────────────────────────────────────────
def categorize_confidence(prob: float) -> str:
    """
    Map a raw sigmoid probability to a human-readable confidence tier.

    Thresholds (symmetric around 0.5):
        High   → prob > 0.80  OR  prob < 0.20
        Medium → 0.50–0.80    OR  0.20–0.50
        Low    → exactly around the decision boundary (0.40–0.60)

    Simplified rule applied by the plan:
        High   > 80 %  → model strongly predicts Cancer
        Medium 50–80 % → moderate confidence
        Low    < 50 %  → model leans Non-Cancer (low cancer probability)

    Args:
        prob: sigmoid output in [0, 1]

    Returns:
        "High" | "Medium" | "Low"
    """
    if prob > 0.80:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    else:
        # Non-cancer prediction — confidence in non-cancer also tiered
        non_cancer_prob = 1.0 - prob
        if non_cancer_prob > 0.80:
            return "High"
        elif non_cancer_prob >= 0.50:
            return "Medium"
        else:
            return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Image validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_image(uploaded_file) -> Image.Image:
    """
    Validate and open a Streamlit UploadedFile as a PIL Image.

    Args:
        uploaded_file: st.file_uploader result object

    Returns:
        PIL.Image

    Raises:
        ValueError: If the file cannot be opened or is not a valid image.
    """
    ALLOWED = {"image/jpeg", "image/png", "image/jpg"}
    if uploaded_file.type not in ALLOWED:
        raise ValueError(
            f"Unsupported file type: {uploaded_file.type}. "
            "Please upload a JPG or PNG image."
        )
    try:
        img = Image.open(io.BytesIO(uploaded_file.read()))
        img.verify()  # check integrity
        # Re-open after verify (verify closes the file pointer)
        img = Image.open(io.BytesIO(uploaded_file.getvalue()))
        return img
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc
