"""
app/app.py
──────────
Cancer Detection System — single-page Streamlit dashboard.

Sections (in order):
  1. Header
  2. Upload + Prediction (2-column)
  3. Explainability — Original + Grad-CAM (2-column)
  4. Model Insights — Loss graph + Metrics (2-column)
  5. Report section
  6. Info section (expander)
  7. Footer / Disclaimer
"""

import os
import sys
import base64
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image as PILImage

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # CancerDetection/
# Ensure ROOT is on path so  model.gradcam  can be imported
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from sibling modules in app/ (Streamlit adds app/ to sys.path)
from predict import load_model, preprocess_image, predict, validate_image as validate_image_format, THRESHOLD
from validator import validate_image as ai_validate_image
from model.gradcam import generate_gradcam
from streamlit_option_menu import option_menu

def get_chart_style():
    # Returns matplotlib style params that match current theme
    return {
        'facecolor': '#0e1117',   # Streamlit dark bg
        'axes_facecolor': '#1a1f2e',
        'text_color': '#ffffff',
        'grid_color': '#2d3748',
        'spine_color': '#2d3748',
    }

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUTS_DIR    = ROOT / "outputs"
TEST_IMAGES_DIR = ROOT / "test_images"

MODEL_OPTIONS = {
    "Phase-1 (Base)":        str(OUTPUTS_DIR / "best_model.h5"),
    "Phase-2 (Fine-tuned)":  str(OUTPUTS_DIR / "best_model_finetuned.h5"),
}

METRICS = {
    "accuracy":  "73.70%",
    "precision": "73.01%",
    "recall":    "75.20%",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Detection System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — card borders, badge colours, typography
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Card wrapper */
    .card {
        background: #ffffff;
        border: 1px solid #e0e4f0;
        border-radius: 12px;
        padding: 20px 22px 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(20,30,80,0.06);
    }

    /* Confidence badges */
    .badge-high   { background:#fee2e2; color:#b91c1c; padding:3px 12px;
                    border-radius:999px; font-weight:600; font-size:13px; }
    .badge-medium { background:#fef9c3; color:#92400e; padding:3px 12px;
                    border-radius:999px; font-weight:600; font-size:13px; }
    .badge-low    { background:#dcfce7; color:#166534; padding:3px 12px;
                    border-radius:999px; font-weight:600; font-size:13px; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 14px;
        padding: 28px 32px 22px 32px;
        margin-bottom: 24px;
    }
    .header-banner h1 { color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700; }
    .header-banner p  { color: #93c5fd; margin: 4px 0 0 0; font-size: 1rem; }

    /* Section headings */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 4px;
        font-style: italic;
    }

    /* Footer */
    .footer-text { color: #64748b; font-size: 0.82rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "uploaded_image":    None,   # PIL.Image
        "prediction_result": None,   # dict
        "gradcam_overlay":   None,   # np.ndarray (BGR)
        "image_source":      None,   # "upload" | "sample" | None
        "pdf_bytes":         None,   # bytes
        "model_choice":      "Phase-2 (Fine-tuned)",
        "image_validation":  None,   # dict from validator.py | None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run prediction + Grad-CAM (cached in session_state)
# ─────────────────────────────────────────────────────────────────────────────
def _run_analysis(pil_image: PILImage.Image, model_path: str):
    """
    Run predict + Grad-CAM and store results in session_state.
    Shows a spinner while working.
    """
    with st.spinner("🔬 Analyzing image…"):
        try:
            model = load_model(model_path)

            # Prediction
            arr = preprocess_image(pil_image)
            result = predict(model, arr)
            st.session_state["prediction_result"] = result

            # Grad-CAM (no disk save from app — pass "" to skip)
            overlay_bgr = generate_gradcam(
                image_input=pil_image,
                model=model,
                output_path="",   # skip disk write
                alpha=0.4,
            )
            st.session_state["gradcam_overlay"] = overlay_bgr

            # Reset old PDF so a fresh one is generated next time
            st.session_state["pdf_bytes"] = None

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.session_state["prediction_result"] = None
            st.session_state["gradcam_overlay"]   = None


# ─────────────────────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    """Section 1 — Header banner + Settings expander."""
    st.markdown(
        """
        <div class="header-banner">
          <h1>🧬 Cancer Detection System</h1>
          <p>Powered by MobileNetV2 &nbsp;·&nbsp; Explainable AI (Grad-CAM)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="
      border: 1px solid #f59e0b;
      border-left: 4px solid #f59e0b;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 16px;
      background-color: var(--secondary-background-color);
      color: var(--text-color);
      font-size: 0.875rem;
    ">
      ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only 
      and is not a substitute for professional medical advice, diagnosis, or 
      treatment. Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Settings — Configure model threshold and display options", expanded=False):
        choice = st.selectbox(
            "Model version",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state["model_choice"]),
            key="_model_selector",
        )
        if choice != st.session_state["model_choice"]:
            # Model changed → clear previous results so they are re-computed
            st.session_state["model_choice"]      = choice
            st.session_state["prediction_result"] = None
            st.session_state["gradcam_overlay"]   = None
            st.session_state["pdf_bytes"]         = None
            st.rerun()

        model_path = MODEL_OPTIONS[choice]
        if Path(model_path).exists():
            st.success(f"✅ Model file found: `{Path(model_path).name}`")
        else:
            st.error(f"⚠️ Model file not found: `{model_path}`")


def render_upload_section() -> bool:
    """
    Section 2 left — Upload card + sample selector.
    Returns True if an image is ready in session_state.
    """
    st.markdown('<div class="section-label">Image Input</div>', unsafe_allow_html=True)

    with st.container():
        # File uploader
        uploaded = st.file_uploader(
            "📤 Upload a histopathology image",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
            help="Upload a 96×96 or larger patch-level histopathology image.",
        )

        if uploaded is not None:
            try:
                pil_img = validate_image_format(uploaded)
                if (
                    st.session_state["image_source"] != "upload"
                    or st.session_state["uploaded_image"] is None
                ):
                    st.session_state["uploaded_image"] = pil_img
                    st.session_state["image_source"]   = "upload"
                    st.session_state["prediction_result"] = None
                    st.session_state["gradcam_overlay"]   = None
                    st.session_state["image_validation"]  = None  # reset
            except ValueError as e:
                st.error(f"❌ {e}")
                return False

        # Sample image selector
        sample_files = sorted(TEST_IMAGES_DIR.glob("*.png")) + sorted(TEST_IMAGES_DIR.glob("*.jpg"))
        sample_names = ["— select a sample —"] + [f.name for f in sample_files]

        selected_sample = st.selectbox(
            "🗂️ Or choose a sample image",
            options=sample_names,
            index=0,
            key="sample_selector",
        )

        if selected_sample != "— select a sample —" and uploaded is None:
            sample_path = TEST_IMAGES_DIR / selected_sample
            if (
                st.session_state["image_source"] != f"sample:{selected_sample}"
                or st.session_state["uploaded_image"] is None
            ):
                st.session_state["uploaded_image"]    = PILImage.open(sample_path).convert("RGB")
                st.session_state["image_source"]      = f"sample:{selected_sample}"
                st.session_state["prediction_result"] = None
                st.session_state["gradcam_overlay"]   = None
                st.session_state["image_validation"]  = None  # reset

        # Image preview
        if st.session_state["uploaded_image"] is not None:
            st.image(
                st.session_state["uploaded_image"],
                caption="Preview",
                width="stretch",
            )

        # Clear button
        if st.session_state["uploaded_image"] is not None:
            if st.button("🗑️ Clear Image", key="clear_btn"):
                for key in ["uploaded_image", "prediction_result",
                            "gradcam_overlay", "image_source", "pdf_bytes"]:
                    st.session_state[key] = None
                st.rerun()

    return st.session_state["uploaded_image"] is not None


def render_prediction_card(prediction: dict):
    """Section 2 right — Prediction result card."""
    st.markdown('<div class="section-label">Model Output</div>', unsafe_allow_html=True)

    label          = prediction["label"]
    conf_pct       = prediction["confidence_pct"]          # cancer %
    non_cancer_pct = prediction.get("non_cancer_pct",
                         round((1 - prediction["probability"]) * 100, 1))
    conf_level     = prediction["confidence_level"]
    prob           = prediction["probability"]

    # Colour by assessment tier
    if "Likely Cancer" in label:
        colour = "#dc2626"   # red
    elif "Uncertain" in label:
        colour = "#d97706"   # amber
    else:
        colour = "#16a34a"   # green

    st.markdown(
        f"<h3 style='color:{colour}; margin-bottom:4px;'>{label}</h3>",
        unsafe_allow_html=True,
    )

    # Confidence badge
    badge_class = f"badge-{conf_level.lower()}"
    st.markdown(
        f"<span class='{badge_class}'>● {conf_level} Confidence</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown(f"""
    <div style="display: flex; gap: 40px; margin: 16px 0;">
      <div>
        <div style="color: #a0a0a0; font-size: 0.8rem; margin-bottom: 4px;">
          🔴 Cancer Probability
        </div>
        <div style="color: #ef4444; font-size: 2.5rem; font-weight: 700;">
          {prob*100:.1f}%
        </div>
      </div>
      <div>
        <div style="color: #a0a0a0; font-size: 0.8rem; margin-bottom: 4px;">
          🟢 Non-Cancer Probability
        </div>
        <div style="color: #22c55e; font-size: 2.5rem; font-weight: 700;">
          {(1-prob)*100:.1f}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dual-color progress bar
    st.markdown(f"""
    <div style="margin: 12px 0 20px 0;">
      <div style="
        height: 12px; 
        border-radius: 6px; 
        background: #22c55e;
        overflow: hidden;
        position: relative;
      ">
        <div style="
          position: absolute; left: 0; top: 0;
          width: {prob*100:.1f}%;
          height: 100%;
          background: #ef4444;
          border-radius: 6px 0 0 6px;
        "></div>
      </div>
      <div style="
        display: flex; 
        justify-content: space-between; 
        margin-top: 4px;
        font-size: 0.75rem; 
        color: #a0a0a0;
      ">
        <span>← Cancer</span>
        <span>Non-Cancer →</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔬 Technical Details"):
        # Dynamic threshold logic boundaries based on app/predict.py
        uncertain_low  = THRESHOLD - 0.15
        uncertain_high = THRESHOLD + 0.20

        # Raw sigmoid detail
        st.caption(
            f"Decision threshold: {THRESHOLD:.2f} (optimized via Youden's J statistic on validation set)<br>"
            f"Raw sigmoid output: **{prob:.4f}** — "
            f"Likely Cancer: **≥{uncertain_high:.2f}** | "
            f"Uncertain: **{uncertain_low:.2f}–{uncertain_high:.2f}** | "
            f"Likely Non-Cancer: **<{uncertain_low:.2f}**",
            unsafe_allow_html=True
        )
    
        # Threshold guide
        st.info(
            f"🔴 **Likely Cancer** (prob ≥ {uncertain_high:.2f})  |  "
            f"🟡 **Uncertain** ({uncertain_low:.2f} – {uncertain_high:.2f})  |  "
            f"🟢 **Likely Non-Cancer** (< {uncertain_low:.2f})"
        )


def render_explainability():
    """Section 3 — Original + Grad-CAM heatmap side-by-side."""
    st.markdown("---")
    st.subheader("🔍 Explainability — Grad-CAM")

    pil_img = st.session_state["uploaded_image"]
    overlay_bgr = st.session_state["gradcam_overlay"]

    # Convert BGR → RGB for display
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption="Original Image", width="stretch")

    with col2:
        st.image(overlay_rgb, caption="Grad-CAM Heatmap", width="stretch")

    st.markdown(
        "<p style='text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 10px;'>"
        "🔴 Red/yellow areas indicate regions the model focused on to make its prediction."
        "</p>",
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style="
        border-left: 4px solid #6366f1;
        padding: 8px 0 8px 16px;
        margin: 24px 0 12px 0;
    ">
        <div style="font-weight: 600; font-size: 1rem; 
        color: var(--text-color);">{title}</div>
        <div style="font-size: 0.8rem; color: #6b7280;">
        {subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_model_insights():
    """Section 4 — Loss curve + Metrics (vertical flow)."""
    st.markdown("---")
    st.subheader("📊 Model Insights")

    # Step 1: Model Performance
    section_header("Model Performance", "Validation metrics from training phase")

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0;">
      <div style="background: var(--secondary-background-color); border: 1px solid #2d3748; border-radius: 8px; padding: 16px;">
        <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Accuracy</div>
        <div style="color: #60a5fa; font-size: 1.8rem; font-weight: 700; margin-top: 4px;">73.70%</div>
      </div>
      <div style="background: var(--secondary-background-color); border: 1px solid #2d3748; border-radius: 8px; padding: 16px;">
        <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Recall</div>
        <div style="color: #34d399; font-size: 1.8rem; font-weight: 700; margin-top: 4px;">75.20%</div>
      </div>
      <div style="background: var(--secondary-background-color); border: 1px solid #2d3748; border-radius: 8px; padding: 16px;">
        <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Precision</div>
        <div style="color: #f472b6; font-size: 1.8rem; font-weight: 700; margin-top: 4px;">73.01%</div>
      </div>
      <div style="background: var(--secondary-background-color); border: 1px solid #2d3748; border-radius: 8px; padding: 16px;">
        <div style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">F1-Score</div>
        <div style="color: #fbbf24; font-size: 1.8rem; font-weight: 700; margin-top: 4px;">74.09%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Step 2: Training Curves
    section_header("Training Curves", "Loss and accuracy over 8 epochs")
    loss_path = OUTPUTS_DIR / "loss.png"
    if loss_path.exists():
        st.image(str(loss_path), caption="Training & Validation Loss", width="stretch")
    else:
        st.warning("⚠️ `outputs/loss.png` not found.")

    st.info(
        "This graph shows how the model minimised binary cross-entropy loss "
        "using gradient descent (∇L). The optimiser adjusts weights in the "
        "direction of steepest descent: **W ← W − α · ∇L**"
    )

    with st.expander("📈 Accuracy Curve"):
        acc_path = OUTPUTS_DIR / "accuracy.png"
        if acc_path.exists():
            st.image(str(acc_path), caption="Training & Validation Accuracy", width="stretch")
        else:
            st.warning("⚠️ `outputs/accuracy.png` not found.")

    # Step 3: Confusion Matrix
    section_header("Confusion Matrix", "Prediction breakdown on validation set")
    with st.expander("🧩 Confusion Matrix"):
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            sns = None
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Transparent backgrounds
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        ax.title.set_color('gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')

        cm = np.array([[14256, 4238], [5512, 16734]])
        
        if sns:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        linewidths=0.5, linecolor='gray', ax=ax,
                        cbar_kws={'label': 'Count'})
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='gray')
            cbar.ax.yaxis.label.set_color('gray')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')
        else:
            cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), horizontalalignment="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
            
        ax.set_title("Confusion Matrix (Validation Set)", fontsize=10)
        tick_marks = np.arange(2) + (0.5 if sns else 0)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(['Non-Cancer', 'Cancer'], fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(['Non-Cancer', 'Cancer'], fontsize=9, rotation=90, va='center')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.legend(framealpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig)


def render_report_section():
    """Section 5 — PDF report generation + download (only after prediction)."""
    st.markdown("---")
    st.subheader("📄 Report")

    pil_img    = st.session_state["uploaded_image"]
    prediction = st.session_state["prediction_result"]
    overlay    = st.session_state["gradcam_overlay"]

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("📄 Generate Report", key="gen_report_btn"):
            with st.spinner("Compiling PDF…"):
                try:
                    from report import generate_pdf_report
                    pdf_bytes = generate_pdf_report(
                        image=pil_img,
                        prediction=prediction,
                        heatmap=overlay,
                        metrics=METRICS,
                    )
                    assert isinstance(pdf_bytes, (bytes, bytearray)), "PDF is not bytes"
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.success("✅ Report generated successfully!")
                except Exception as exc:
                    st.error(f"❌ Could not generate report: {exc}")

    if st.session_state.get("pdf_bytes"):
        with col2:
            st.download_button(
                label="📥 Download PDF",
                data=st.session_state["pdf_bytes"],
                file_name="cancer_detection_report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )


def render_info_section():
    """Section 6 — Collapsible About expander (always visible)."""
    st.markdown("---")

    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown(
            """
### What is IDC?
**Invasive Ductal Carcinoma (IDC)** is the most common form of breast cancer.
It originates in the milk ducts and invades surrounding tissue. Early detection
through microscopic tissue analysis is critical for improving patient outcomes.
This model classifies 96×96 pixel histopathology image patches as IDC-positive
(cancer) or IDC-negative (non-cancer).

---

### How the Model Works
The system uses **MobileNetV2**, a lightweight convolutional neural network
pre-trained on ImageNet.

- The base convolutional layers are **frozen** — they act as a universal image
  feature extractor.
- A custom classification head is added (GlobalAveragePooling → Dense 128 →
  Dropout 0.5 → Dense 1 with sigmoid).
- Only the head (and fine-tuned final layers in Phase-2) are trained on the
  histopathology dataset.

This technique is called **transfer learning** — it allows effective training
with limited data by reusing learned visual features.

---

### Training Method
This model is trained on patch-level histopathology images using transfer
learning. Training used binary cross-entropy loss, the Adam optimiser,
early stopping, and model checkpointing.  Phase-2 fine-tuning unlocked the
last 20 layers of MobileNetV2 with a reduced learning rate for additional
accuracy gains.

---

### Dataset
- **Source:** Kaggle — Breast Histopathology Images (IDC)
- **Size:** ~277,524 patches from 162 whole-mount slide images
- **Split:** ~70% train / 15% validation / 15% test

---

### Limitations
- Patch-level classification only (not whole-slide diagnosis)
- Not validated for clinical or diagnostic use
- For educational and research purposes only
            """
        )


def render_footer():
    """Section 7 — Disclaimer footer."""
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This system is not a medical diagnosis tool. "
        "Predictions are produced by a machine-learning model and carry no "
        "clinical validity. For educational and research purposes only. "
        "Always consult a qualified medical professional."
    )
    st.markdown(
        "<div class='footer-text'>Cancer Detection System · MobileNetV2 · "
        "Grad-CAM · Streamlit · Built for educational use</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _init_state()

    # ── Section 1: Header ────────────────────────────────────────────────────
    render_header()

    selected = option_menu(
        menu_title=None,
        options=["Detect", "Model Insights", "Report", "About"],
        icons=["search", "bar-chart-line", "file-earmark-text", "info-circle"],
        orientation="horizontal",
        styles={
            "container": {
                "padding": "4px",
                "background-color": "transparent",
                "border": "1px solid rgba(255,255,255,0.1)",
                "border-radius": "10px",
            },
            "icon": {
                "font-size": "14px",
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "padding": "10px 20px",
                "border-radius": "8px",
                "background-color": "transparent",
                "color": "var(--text-color)",
            },
            "nav-link-hover": {
                "background-color": "rgba(99, 102, 241, 0.15)",
            },
            "nav-link-selected": {
                "background-color": "#6366f1",
                "color": "white",
                "font-weight": "600",
            },
        }
    )

    st.markdown("""
    <style>
      /* Fix option menu container background */
      nav[data-testid="stHorizontalBlock"],
      .nav-link, ul[class*="nav"] {
        background-color: transparent !important;
      }
      
      /* Fix the white box around the whole menu */
      div[data-testid="stHorizontalBlock"] > div {
        background-color: transparent !important;
      }

      /* Fix upload button color in dark mode */
      [data-testid="stFileUploaderDropzone"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
      }

      /* Fix all expander headers */
      [data-testid="stExpander"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
      }
    </style>
    """, unsafe_allow_html=True)

    if selected == "Detect":
        st.markdown("### 🔬 Cancer Detection")
        st.caption("Upload a histopathology image to analyze for IDC cancer")

        # ── Section 2: Upload + Prediction ───────────────────────────────────────
        col_upload, col_pred = st.columns([1, 1], gap="large")

        with col_upload:
            image_ready = render_upload_section()

        # Auto-trigger validation + analysis when image is ready
        if image_ready and st.session_state["prediction_result"] is None:

            # ── Step A: AI image validation (cached per image) ─────────────────
            if st.session_state["image_validation"] is None:
                with st.spinner("🔍 Validating image…"):
                    api_key = None
                    try:
                        api_key = st.secrets.get("AI_API_KEY", None)
                    except Exception:
                        pass
                    st.session_state["image_validation"] = ai_validate_image(
                        st.session_state["uploaded_image"], api_key
                    )

            val = st.session_state["image_validation"]

            if val["skipped"]:
                # API unavailable — show warning but proceed
                st.warning(
                    f"⚠️ Validation unavailable — proceeding with prediction. "
                    f"({val['label']})"
                )
            elif not val["is_valid"]:
                # Definitely not a medical image — stop here
                with col_pred:
                    st.error(
                        f"❌ Invalid input detected: **{val['label']}**  "
                        f"(confidence {val['confidence']:.0%}).\n\n"
                        "Please upload a **medical histopathology image**."
                    )
                return  # do not run analysis

            # ── Step B: Run ML analysis ─────────────────────────────────────────
            model_path = MODEL_OPTIONS[st.session_state["model_choice"]]
            _run_analysis(st.session_state["uploaded_image"], model_path)
            
            # Map prediction session var natively
            if st.session_state["prediction_result"] is not None:
                st.session_state['prediction'] = {
                    'cancer_prob': st.session_state["prediction_result"]["probability"],
                    'label': st.session_state["prediction_result"]["label"],
                    'confidence': st.session_state["prediction_result"]["confidence_pct"],
                    'image': st.session_state["uploaded_image"]
                }
            st.rerun()

        with col_pred:
            # Show validation badge if a result exists
            val = st.session_state.get("image_validation")
            if val and not val["skipped"] and val["is_valid"]:
                st.success(
                    f"✅ Image validated as medical image "
                    f"({val['confidence']:.0%} confidence)"
                )

            if st.session_state["prediction_result"] is not None:
                render_prediction_card(st.session_state["prediction_result"])
            elif st.session_state["uploaded_image"] is not None:
                # Image uploaded but analysis still running (first pass)
                st.info("🔄 Running analysis…")
            else:
                st.markdown(
                    "<div style='color:#94a3b8; padding-top:40px; text-align:center;'>"
                    "Upload or select an image to see the prediction.</div>",
                    unsafe_allow_html=True,
                )

        # ── Section 3: Explainability ─────────────────────────────────────────────
        if (
            st.session_state["prediction_result"] is not None
            and st.session_state["gradcam_overlay"] is not None
        ):
            render_explainability()

    elif selected == "Model Insights":
        st.markdown("### 📊 Model Insights")
        st.caption("Training metrics and performance evaluation")
        render_model_insights()

    elif selected == "Report":
        st.markdown("### 📄 Report")
        st.caption("Generate and download a PDF report of the last prediction")
        if st.session_state.get("prediction_result") is None:
            st.info("No prediction yet. Go to the Detect tab and upload an image first.")
        else:
            render_report_section()

    elif selected == "About":
        st.markdown("### ℹ️ About This System")
        st.caption("How the model works, dataset info, and limitations")
        render_info_section()
        render_footer()

if __name__ == "__main__":
    main()
