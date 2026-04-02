"""
app/report.py
─────────────
PDF report generator for the Cancer Detection dashboard.

Uses fpdf2 (pip install fpdf2) to build a multi-page PDF entirely in memory
and return it as bytes so Streamlit can offer a download button.

Pages
-----
1. Title page
2. Uploaded image + prediction result + confidence
3. Grad-CAM side-by-side (original + heatmap overlay)
4. Model performance metrics table
5. Math appendix — gradient descent note
6. Disclaimer
"""

import io
import datetime
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from fpdf import FPDF
except ImportError as e:
    raise ImportError(
        "fpdf2 is required for PDF generation. "
        "Install it with: pip install fpdf2"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_temp_png(pil_or_array, suffix: str = "") -> str:
    """Save a PIL Image or numpy BGR array to a temp PNG; return the path."""
    if isinstance(pil_or_array, np.ndarray):
        # Assume BGR from OpenCV — convert to RGB PIL
        rgb = cv2.cvtColor(pil_or_array, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(rgb)
    else:
        img = pil_or_array.convert("RGB")

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"{suffix}.png", delete=False
    )
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def _safe(text: str) -> str:
    """Strip / replace characters outside Latin-1 so Helvetica doesn't crash."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _heading(pdf: "FPDF", text: str, size: int = 14, color=(30, 30, 80)):
    """Render a section heading."""
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, _safe(text), ln=True)
    pdf.set_text_color(0, 0, 0)


def _body(pdf: "FPDF", text: str, size: int = 10):
    """Render a body paragraph."""
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(0, 6, _safe(text))


def _divider(pdf: "FPDF"):
    pdf.ln(3)
    pdf.set_draw_color(180, 180, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


# ─────────────────────────────────────────────────────────────────────────────
# Private page builders
# ─────────────────────────────────────────────────────────────────────────────

def _add_title_page(pdf: "FPDF"):
    """Page 1 — Title banner."""
    pdf.add_page()
    pdf.set_fill_color(20, 30, 70)
    pdf.rect(0, 0, pdf.w, 60, "F")

    pdf.set_y(15)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Cancer Detection System", ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Powered by MobileNetV2 + Explainable AI (Grad-CAM)", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.ln(30)
    pdf.set_font("Helvetica", "", 10)
    date_str = datetime.datetime.now().strftime("%B %d, %Y  |  %H:%M")
    pdf.cell(0, 6, _safe(f"Report generated: {date_str}"), ln=True, align="C")

    pdf.ln(6)
    _body(
        pdf,
        "This report was produced automatically by the AI-based cancer "
        "detection web application. It contains the uploaded histopathology "
        "image, the model prediction, explainability heatmap, and "
        "performance metrics. See the disclaimer on the final page.",
    )


def _add_prediction_page(pdf: "FPDF", image: PILImage.Image, prediction: dict):
    """Page 2 — Uploaded image + prediction result."""
    pdf.add_page()
    _heading(pdf, "Prediction Result", 16)
    _divider(pdf)

    # Save image as temp file so FPDF can load it
    img_path = _pil_to_temp_png(image, suffix="_upload")
    try:
        pdf.image(img_path, x=15, y=pdf.get_y(), w=90)
        pdf.ln(95)
    finally:
        Path(img_path).unlink(missing_ok=True)

    # Prediction fields
    label = prediction.get("label", "N/A")
    confidence_pct = prediction.get("confidence_pct", 0.0)
    confidence_level = prediction.get("confidence_level", "N/A")
    probability = prediction.get("probability", 0.0)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 7, "Prediction:", ln=False)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, label, ln=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 7, "Raw Probability:", ln=False)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, f"{probability:.4f}", ln=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 7, "Confidence Score:", ln=False)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, f"{confidence_pct}%", ln=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(60, 7, "Confidence Level:", ln=False)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, confidence_level, ln=True)

    pdf.ln(4)
    _body(
        pdf,
        "Note: These results are produced by a deep-learning model trained "
        "on patch-level histopathology images. They are not a clinical "
        "diagnosis and should not replace professional medical evaluation.",
    )


def _add_gradcam_page(pdf: "FPDF", original: PILImage.Image, overlay: np.ndarray):
    """Page 3 — Original image + Grad-CAM heatmap side-by-side."""
    pdf.add_page()
    _heading(pdf, "Grad-CAM Explainability", 16)
    _divider(pdf)

    _body(
        pdf,
        "The Grad-CAM heatmap highlights the image regions that most "
        "strongly influenced the model's prediction. Red/yellow areas "
        "indicate high gradient activations; blue areas had little effect.",
    )
    pdf.ln(4)

    # Save both images as temp files
    orig_path = _pil_to_temp_png(original, suffix="_orig")
    heat_path = _pil_to_temp_png(overlay, suffix="_heat")
    try:
        y = pdf.get_y()
        pdf.image(orig_path, x=15, y=y, w=85)
        pdf.image(heat_path, x=110, y=y, w=85)
        pdf.ln(90)
    finally:
        Path(orig_path).unlink(missing_ok=True)
        Path(heat_path).unlink(missing_ok=True)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(95, 5, "Original Image", align="C", ln=False)
    pdf.cell(95, 5, "Grad-CAM Overlay", align="C", ln=True)
    pdf.set_text_color(0, 0, 0)


def _add_metrics_page(pdf: "FPDF", metrics: dict):
    """Page 4 — Model performance metrics table."""
    pdf.add_page()
    _heading(pdf, "Model Performance Metrics", 16)
    _divider(pdf)

    _body(
        pdf,
        "The metrics below were recorded on the validation set during "
        "training (Phase-2 fine-tuning). They reflect overall model "
        "performance and are NOT specific to the uploaded image.",
    )
    pdf.ln(6)

    # Table header
    pdf.set_fill_color(30, 30, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    col_w = [80, 80]
    pdf.cell(col_w[0], 9, "Metric", border=1, fill=True)
    pdf.cell(col_w[1], 9, "Value", border=1, fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)

    # Table rows
    rows = [
        ("Accuracy",  metrics.get("accuracy",  "73.70%")),
        ("Precision", metrics.get("precision", "73.01%")),
        ("Recall",    metrics.get("recall",    "75.20%")),
    ]
    pdf.set_font("Helvetica", "", 11)
    for i, (name, val) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(235, 235, 250) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w[0], 8, _safe(name), border=1, fill=True)
        pdf.cell(col_w[1], 8, _safe(str(val)), border=1, fill=True, ln=True)


def _add_math_appendix(pdf: "FPDF"):
    """Page 5 — Gradient descent math note."""
    pdf.add_page()
    _heading(pdf, "Appendix: Gradient Descent & Loss Minimization", 16)
    _divider(pdf)

    sections = [
        (
            "Binary Cross-Entropy Loss",
            "The model is optimised by minimising the binary cross-entropy "
            "loss function:\n\n"
            "    L(y, y') = -[ y * log(y') + (1-y) * log(1-y') ]\n\n"
            "where y is the true label (0 or 1) and y' is the sigmoid output "
            "of the final layer.",
        ),
        (
            "Gradient Descent",
            "At each training step the optimizer computes the gradient of the "
            "loss with respect to every trainable weight W:\n\n"
            "    grad_W L  =  dL / dW\n\n"
            "and updates the weights in the direction that decreases the loss:\n\n"
            "    W  <-  W  -  (learning_rate) * grad_W L\n\n"
            "This iterative process is called Stochastic Gradient Descent "
            "(SGD) or, in this project, the Adam adaptive variant.",
        ),
        (
            "Connection to Vector Calculus",
            "The gradient vector grad_W L lives in weight-space; each "
            "component is the partial derivative of L with respect to one "
            "individual weight.  Moving opposite to this vector is the "
            "direction of steepest descent on the loss surface.  Grad-CAM "
            "exploits the same idea: it computes the gradient of the class "
            "score with respect to the feature maps of the final "
            "convolutional layer to find which spatial regions matter most.",
        ),
    ]

    for title, body in sections:
        _heading(pdf, title, 12, color=(20, 60, 140))
        _body(pdf, body)
        pdf.ln(3)


def _add_disclaimer(pdf: "FPDF"):
    """Final page — disclaimer."""
    pdf.add_page()
    pdf.set_fill_color(255, 243, 205)
    pdf.rect(10, 10, pdf.w - 20, 70, "F")
    pdf.set_xy(15, 15)
    _heading(pdf, "DISCLAIMER", 14, color=(180, 90, 0))
    pdf.set_xy(15, 27)
    _body(
        pdf,
        "This system is NOT a medical diagnosis tool.\n\n"
        "It is intended exclusively for educational and research purposes. "
        "The predictions produced by this model must not be used as a basis "
        "for any clinical or medical decision. Always consult a qualified "
        "medical professional for diagnosis and treatment.\n\n"
        "The model was trained on a publicly available histopathology "
        "dataset and has not been validated for clinical use.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    image: PILImage.Image,
    prediction: dict,
    heatmap: np.ndarray,
    metrics: dict | None = None,
) -> bytes:
    """
    Compile a multi-page PDF report and return it as raw bytes.

    Args:
        image:      PIL Image that was uploaded / selected.
        prediction: Dict from predict.predict() —
                    keys: probability, label, confidence_pct, confidence_level
        heatmap:    BGR numpy array from generate_gradcam().
        metrics:    Optional dict with keys "accuracy", "precision", "recall".
                    Defaults to Phase-2 training metrics if not supplied.

    Returns:
        bytes: PDF file contents ready to pass to st.download_button.
    """
    if metrics is None:
        metrics = {
            "accuracy":  "73.70%",
            "precision": "73.01%",
            "recall":    "75.20%",
        }

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)

    _add_title_page(pdf)
    _add_prediction_page(pdf, image, prediction)
    _add_gradcam_page(pdf, image, heatmap)
    _add_metrics_page(pdf, metrics)
    _add_math_appendix(pdf)
    _add_disclaimer(pdf)

    pdf_output = pdf.output(dest='S')

    # CRITICAL FIX
    if isinstance(pdf_output, bytearray):
        pdf_bytes = bytes(pdf_output)
    elif isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode('latin-1')
    else:
        raise TypeError(f"Unexpected PDF output type: {type(pdf_output)}")

    return pdf_bytes
