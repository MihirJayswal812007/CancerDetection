import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union
from PIL import Image as PILImage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model


def generate_gradcam(
    image_input: Union[str, "PILImage.Image", np.ndarray],
    model: Model,
    output_path: str = "outputs/heatmap.png",
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap and overlays it on the input image.

    Compatible with TF 2.15 – 2.21+.  Avoids constructing a secondary
    functional Model (which breaks across TF versions when MobileNetV2 is
    stored as a nested sub-model).  Instead it runs a custom forward pass
    via the sub-model reference, capturing the feature maps directly.

    Args:
        image_input: File path (str/Path), PIL.Image, or numpy array (H,W,3).
                     Accepts str for backward compatibility with standalone tests.
        model:       Trained binary classifier (MobileNetV2 + sigmoid head).
        output_path: Where to save the overlay PNG.  Pass "" to skip saving.
        alpha:       Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        np.ndarray: BGR overlay image (224×224×3).
    """
    IMG_SIZE = (224, 224)

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    # Accept file path (str/Path), PIL Image, or numpy array
    if isinstance(image_input, (str, Path)):
        original_pil = load_img(str(image_input), target_size=IMG_SIZE)
        img_array    = img_to_array(original_pil) / 255.0
    elif isinstance(image_input, PILImage.Image):
        original_pil = image_input.convert("RGB").resize(IMG_SIZE)
        img_array    = np.array(original_pil, dtype=np.float32) / 255.0
    elif isinstance(image_input, np.ndarray):
        # Expect (H, W, 3) uint8 or float32
        pil_tmp      = PILImage.fromarray(
            (image_input * 255).astype(np.uint8)
            if image_input.max() <= 1.0
            else image_input.astype(np.uint8)
        ).resize(IMG_SIZE)
        original_pil = pil_tmp
        img_array    = np.array(pil_tmp, dtype=np.float32) / 255.0
    else:
        raise TypeError(
            f"image_input must be a file path, PIL.Image, or np.ndarray. "
            f"Got {type(image_input)}."
        )
    img_batch = tf.constant(
        np.expand_dims(img_array, axis=0), dtype=tf.float32
    )

    # ── 2. Locate the nested MobileNetV2 sub-model ───────────────────────────
    # train.py wraps MobileNetV2 as a nested Model layer.
    # We call it separately to get the 7×7×1280 feature maps.
    base_model = None
    classifier_layers = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Model):
            base_model = layer
        else:
            if base_model is not None:
                # Everything after the base model = the custom head
                classifier_layers.append(layer)

    if base_model is None:
        raise ValueError(
            "Could not find nested base model. "
            "Expected model structure: Input → MobileNetV2 → head layers."
        )

    print(f"  Base model            : {base_model.name}")
    print(f"  Head layers           : {[l.name for l in classifier_layers]}")

    # ── 3. Run forward pass, capturing conv output + final prediction ────────
    with tf.GradientTape() as tape:
        # Step A: through base model → (1, 7, 7, 1280)
        conv_outputs = base_model(img_batch, training=False)
        tape.watch(conv_outputs)

        # Step B: through the custom head manually
        x = conv_outputs
        for layer in classifier_layers:
            x = layer(x, training=False)

        score = x[:, 0]   # cancer probability (sigmoid output)

    # ── 4. Gradients of score w.r.t. feature maps ────────────────────────────
    grads = tape.gradient(score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "GradientTape returned None. "
            "Ensure the head layers are called inside the tape context."
        )

    # ── 5. Global-average pool gradients → channel importance weights ────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()   # (C,)

    # ── 6. Weighted sum over channels → scalar heatmap ───────────────────────
    feat_maps = conv_outputs[0].numpy()          # (7, 7, 1280)
    heatmap   = np.dot(feat_maps, pooled_grads)  # (7, 7)

    # ── 7. ReLU + normalise to [0, 1] ────────────────────────────────────────
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # ── 8. Resize & apply JET colormap ───────────────────────────────────────
    heatmap_uint8   = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # ── 9. Load base image as BGR ─────────────────────────────────────────────
    if isinstance(image_input, (str, Path)) and Path(str(image_input)).is_file():
        base_bgr = cv2.imread(str(image_input))
    else:
        base_bgr = None
    if base_bgr is None:
        base_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    base_bgr = cv2.resize(base_bgr, IMG_SIZE)

    # ── 10. Overlay: base × (1−alpha) + heatmap × alpha ──────────────────────
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    # ── 11. Save (skip if output_path is empty string) ────────────────────────
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Standalone diagnostic test  →  python model/gradcam.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Grad-CAM Diagnostic Test")
    print("=" * 60)

    MODEL_PATH  = Path("outputs/best_model.h5")
    SEARCH_DIRS = [Path("dataset/cancer"), Path("dataset/non-cancer")]

    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")

    test_images = []
    for d in SEARCH_DIRS:
        if d.exists():
            imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            if imgs:
                test_images.append(imgs[0])
        if len(test_images) == 2:
            break

    if not test_images:
        sys.exit("No test images. Add .png/.jpg to dataset/cancer/ or dataset/non-cancer/")

    print(f"\n  Loading model : {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"  Top-level layers: {[l.name for l in model.layers]}")

    for i, img_path in enumerate(test_images, 1):
        out = Path(f"outputs/test_heatmap_{i}.png")
        print(f"\n  -- Test {i}: {img_path.name} --")
        result = generate_gradcam(img_path, model, output_path=str(out))

        assert result is not None,            "FAIL: returned None"
        assert result.shape == (224, 224, 3), f"FAIL: shape {result.shape}"
        assert result.max() > 0,              "FAIL: overlay is blank (all zeros)"

        print(f"  Output shape  : {result.shape}")
        print(f"  Pixel range   : [{result.min()}, {result.max()}]")
        print(f"  Saved to      : {out}")
        print(f"  PASS")

    print(f"\n  All {len(test_images)} test(s) passed.")
    print(f"  Open outputs/test_heatmap_*.png to verify visually.")
    print("=" * 60)
