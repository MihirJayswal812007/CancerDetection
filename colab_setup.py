# =============================================================================
# CANCER DETECTION — COLAB SETUP & TRAINING NOTEBOOK
# =============================================================================
# Instructions: Paste each section into a separate Colab cell in order.
# Runtime -> Change runtime type -> GPU (T4) BEFORE running.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — GPU CHECK
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ GPU detected")
    print(result.stdout.split("\n")[8])  # GPU name line
else:
    raise RuntimeError(
        "❌ No GPU found. Go to Runtime → Change runtime type → GPU and restart."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

packages = ["kaggle", "tensorflow", "opencv-python-headless", "matplotlib", "pillow"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✅ Dependencies installed")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — MOUNT GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/cancer-detection")
DRIVE_DATASET = DRIVE_ROOT / "dataset"
DRIVE_OUTPUT  = DRIVE_ROOT / "outputs"

# Create folder skeleton on Drive
for folder in [
    DRIVE_DATASET / "cancer",
    DRIVE_DATASET / "non-cancer",
    DRIVE_OUTPUT,
]:
    folder.mkdir(parents=True, exist_ok=True)

print(f"✅ Drive mounted. Project root: {DRIVE_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — KAGGLE AUTHENTICATION
#
# TWO OPTIONS — choose one and delete the other:
#
# OPTION A (Recommended): Upload your kaggle.json via the Colab file picker
# OPTION B: Hard-code credentials (less secure, fine for short sessions)
# ─────────────────────────────────────────────────────────────────────────────

# === OPTION A: Upload kaggle.json ===
# Step 1: Get your token at https://www.kaggle.com/settings -> API -> Create New Token
# Step 2: Run the cell below, click "Choose Files", select kaggle.json

from google.colab import files
import os, json, stat

print("Upload your kaggle.json file now ↓")
uploaded = files.upload()  # Interactive file picker

kaggle_dir = Path("/root/.kaggle")
kaggle_dir.mkdir(exist_ok=True)
kaggle_json_path = kaggle_dir / "kaggle.json"

for filename, content in uploaded.items():
    kaggle_json_path.write_bytes(content)

# Validate JSON
creds = json.loads(kaggle_json_path.read_text())
assert "username" in creds and "key" in creds, \
    "❌ kaggle.json must contain 'username' and 'key' fields"

# Restrict permissions (required by Kaggle CLI)
os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)

print(f"✅ Kaggle credentials saved for user: {creds['username']}")


# === OPTION B: Hard-code credentials (uncomment if not using Option A) ===
# import os, stat
# from pathlib import Path

# KAGGLE_USERNAME = "your_username_here"   # ← replace
# KAGGLE_KEY      = "your_api_key_here"    # ← replace

# kaggle_dir = Path("/root/.kaggle")
# kaggle_dir.mkdir(exist_ok=True)
# kaggle_json_path = kaggle_dir / "kaggle.json"
# kaggle_json_path.write_text(
#     f'{{"username": "{KAGGLE_USERNAME}", "key": "{KAGGLE_KEY}"}}'
# )
# os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)
# print(f"✅ Kaggle credentials set for: {KAGGLE_USERNAME}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — DOWNLOAD DATASET FROM KAGGLE
#
# Dataset: Breast Histopathology Images (PatchCamelyon-style)
# URL: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
# Size: ~2.7 GB unzipped | ~162K images (IDC patches)
# Classes: 0 = non-cancerous, 1 = cancerous (Invasive Ductal Carcinoma)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess

RAW_DOWNLOAD_DIR = Path("/content/raw_kaggle")
RAW_DOWNLOAD_DIR.mkdir(exist_ok=True)

print("⬇️  Downloading dataset... (this may take 5–10 minutes on Colab)")
result = subprocess.run(
    [
        "kaggle", "datasets", "download",
        "-d", "paultimothymooney/breast-histopathology-images",
        "-p", str(RAW_DOWNLOAD_DIR),
        "--unzip",
    ],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("STDERR:", result.stderr)
    raise RuntimeError("❌ Kaggle download failed. Check credentials and dataset name.")

print("✅ Download complete")
print(f"Raw files in: {RAW_DOWNLOAD_DIR}")

# Quick count
all_images = list(RAW_DOWNLOAD_DIR.rglob("*.png"))
print(f"   Total images found: {len(all_images):,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — ORGANIZE INTO cancer/ AND non-cancer/ STRUCTURE
#
# The Kaggle dataset stores images in per-patient folders named:
#   <patient_id>/0/*.png  → non-cancerous (IDC negative)
#   <patient_id>/1/*.png  → cancerous     (IDC positive)
#
# We copy up to MAX_PER_CLASS images from each label folder.
# Default: 5,000 per class (10,000 total) — good balance of speed and accuracy.
# Increase to 20,000+ per class for production-quality training.
# ─────────────────────────────────────────────────────────────────────────────
import shutil
from pathlib import Path

MAX_PER_CLASS = 5_000  # ← Adjust: more = better accuracy but slower training

cancer_dir     = DRIVE_DATASET / "cancer"
noncancer_dir  = DRIVE_DATASET / "non-cancer"

# Only copy if destination is empty (resume-safe)
cancer_existing    = len(list(cancer_dir.iterdir()))
noncancer_existing = len(list(noncancer_dir.iterdir()))

if cancer_existing >= MAX_PER_CLASS and noncancer_existing >= MAX_PER_CLASS:
    print(f"✅ Dataset already organized: {cancer_existing} cancer, {noncancer_existing} non-cancer images")
else:
    print(f"📦 Organizing images (up to {MAX_PER_CLASS:,} per class)...")
    cancer_count    = cancer_existing
    noncancer_count = noncancer_existing

    for img_path in RAW_DOWNLOAD_DIR.rglob("*.png"):
        label_dir_name = img_path.parent.name  # "0" or "1"

        if label_dir_name == "1" and cancer_count < MAX_PER_CLASS:
            shutil.copy2(img_path, cancer_dir / img_path.name)
            cancer_count += 1

        elif label_dir_name == "0" and noncancer_count < MAX_PER_CLASS:
            shutil.copy2(img_path, noncancer_dir / img_path.name)
            noncancer_count += 1

        if cancer_count >= MAX_PER_CLASS and noncancer_count >= MAX_PER_CLASS:
            break

    print(f"✅ Organization complete:")
    print(f"   cancer/     → {cancer_count:,} images")
    print(f"   non-cancer/ → {noncancer_count:,} images")

# Verify structure
assert len(list(cancer_dir.iterdir())) > 0,    "❌ cancer/ folder is still empty"
assert len(list(noncancer_dir.iterdir())) > 0, "❌ non-cancer/ folder is still empty"
print("✅ Dataset structure verified")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — CLONE / COPY PROJECT CODE FROM DRIVE
#
# Option A (recommended): Your project lives in Drive already
# Option B: Clone from GitHub
# ─────────────────────────────────────────────────────────────────────────────

# === OPTION A: Project already in Drive (standard workflow) ===
import os
os.chdir(str(DRIVE_ROOT))
print(f"✅ Working directory set to: {os.getcwd()}")

# Verify model scripts are present
for required in ["model/utils.py", "model/train.py"]:
    assert Path(required).exists(), \
        f"❌ Missing {required} — copy your project to {DRIVE_ROOT}"
print("✅ Source files found")


# === OPTION B: Clone from GitHub (uncomment if using version control) ===
# import subprocess
# subprocess.run(
#     ["git", "clone", "https://github.com/YOUR_USERNAME/CancerDetection.git",
#      str(DRIVE_ROOT)],
#     check=True
# )
# os.chdir(str(DRIVE_ROOT))
# print(f"✅ Repo cloned to {DRIVE_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — VALIDATE DATA PIPELINE (smoke test before full training)
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(DRIVE_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — IMPROVED AUGMENTATION
# Replace existing ImageDataGenerator with this
# ─────────────────────────────────────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 96     # keep existing size
BATCH_SIZE = 32     # keep existing batch size

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],    # NEW: brightness variation
    channel_shift_range=10.0,        # NEW: slight color noise
    fill_mode='nearest',
    validation_split=0.2
)

# Validation: ONLY rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    str(DRIVE_DATASET),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    str(DRIVE_DATASET),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=42
)

print(f"✅ Augmented train generator: {train_gen.samples:,} images")
print(f"✅ Clean val generator      : {val_gen.samples:,} images")
\n\nfrom model.utils import create_image_generators

print("🔍 Validating data pipeline...")
train_gen, val_gen = create_image_generators(
    dataset_path=str(DRIVE_DATASET),
    image_size=(224, 224),
    batch_size=32,
)

print(f"\n✅ Generators created successfully")
print(f"   Train samples       : {train_gen.samples:,}")
print(f"   Validation samples  : {val_gen.samples:,}")
print(f"   Class indices       : {train_gen.class_indices}")
print(f"   Expected mapping    : {{'cancer': 0, 'non-cancer': 1}}")

# Pull one batch to confirm shapes
batch_x, batch_y = next(train_gen)
print(f"\n   Batch image shape   : {batch_x.shape}")   # Expected: (32, 224, 224, 3)
print(f"   Batch label shape   : {batch_y.shape}")   # Expected: (32,)
print(f"   Pixel value range   : [{batch_x.min():.3f}, {batch_x.max():.3f}]")  # Expect ~[0, 1]

assert batch_x.shape == (32, 224, 224, 3), "Unexpected image batch shape"
assert batch_x.max() <= 1.0,              "Normalization failed — values > 1.0"
print("\n✅ Data pipeline smoke test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — BUILD MODEL AND VERIFY ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
from model.train import build_model, get_parameter_counts

print("🔍 Building model...")
model = build_model(input_shape=(224, 224, 3))

counts = get_parameter_counts(model)
print(f"\n✅ Model built: {model.name}")
print(f"   Trainable params     : {counts['trainable']:,}")
print(f"   Non-trainable params : {counts['non_trainable']:,}")
print(f"   Total params         : {counts['total']:,}")

# Verify frozen backbone
for layer in model.layers:
    if "mobilenet" in layer.name.lower():
        assert not layer.trainable, "❌ MobileNetV2 backbone should be frozen"
print("   Backbone frozen      : ✅")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
from pathlib import Path

EPOCHS = 10

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(DRIVE_OUTPUT / "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

print(f"🚀 Starting training for up to {EPOCHS} epochs...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)
print("✅ Training complete")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — SAVE ARTIFACTS (model + plots)
# ─────────────────────────────────────────────────────────────────────────────
from model.train import save_training_artifacts

save_training_artifacts(model, history, DRIVE_OUTPUT)

print("✅ Artifacts saved to Drive:")
for f in sorted(DRIVE_OUTPUT.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"   {f.name:25s} {size_kb:>8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — VALIDATION CHECKS (training quality)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab
import matplotlib.pyplot as plt

h = history.history
epochs_ran = len(h["loss"])

print("=" * 50)
print("TRAINING RESULTS SUMMARY")
print("=" * 50)
print(f"Epochs completed      : {epochs_ran}")
print(f"Final train loss      : {h['loss'][-1]:.4f}")
print(f"Final val loss        : {h['val_loss'][-1]:.4f}")
print(f"Final train accuracy  : {h['accuracy'][-1]:.4f}")
print(f"Final val accuracy    : {h['val_accuracy'][-1]:.4f}")
print(f"Final precision       : {h['precision'][-1]:.4f}")
print(f"Final recall          : {h['recall'][-1]:.4f}")

# Check 1: Loss decreasing
loss_decreased = h["loss"][0] > h["loss"][-1]
print(f"\n✅ Loss decreased      : {loss_decreased}")

# Check 2: No severe overfitting (val_loss within 20% of train_loss)
overfit_ratio = h["val_loss"][-1] / max(h["loss"][-1], 1e-8)
overfitting   = overfit_ratio > 1.5
print(f"   Val/Train loss ratio: {overfit_ratio:.2f}  {'⚠️  Overfitting detected' if overfitting else '✅ OK'}")

# Check 3: Accuracy above random chance (>60%)
accuracy_ok = h["val_accuracy"][-1] > 0.60
print(f"   Val accuracy > 60%  : {'✅' if accuracy_ok else '⚠️  Low — check dataset balance'}")

# Display plots inline
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(h["loss"],     label="Train Loss",        linewidth=2)
axes[0].plot(h["val_loss"], label="Validation Loss",   linewidth=2, linestyle="--")
axes[0].set_title("Loss Curve (Gradient Descent Behaviour)", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(h["accuracy"],     label="Train Accuracy",      linewidth=2)
axes[1].plot(h["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
axes[1].set_title("Accuracy Curve", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ Validation checks complete")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — FINE-TUNING  (unfreeze last 20 MobileNetV2 layers, lr=1e-5)
#
# Run this cell AFTER Cell 12.  The best model from Phase-1 training is loaded
# from Drive so you always start from the strongest checkpoint.
#
# Strategy
# ─────────
#  • Keep layers [0 … N-21] of MobileNetV2 frozen (low-level feature detectors)
#  • Unfreeze layers [N-20 … N]  (high-level, domain-specific features)
#  • Use a very small lr (1e-5) to avoid destroying pre-trained weights
#  • Same callbacks as Phase-1 (EarlyStopping / ModelCheckpoint / ReduceLROnPlateau)
#  • Save as best_model_finetuned.h5 so Phase-1 weights are never overwritten
# ─────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
from pathlib import Path

# ── 0. Record Phase-1 baseline ────────────────────────────────────────────────
try:
    phase1_val_acc = h["val_accuracy"][-1]
except NameError:
    print("⚠️ 'h' (history) not found in memory. Using hardcoded Phase-1 baseline (73.55%).")
    phase1_val_acc = 0.7355

print("=" * 55)
print("  FINE-TUNING  —  Phase 2")
print("=" * 55)
print(f"\n  Phase-1 best val accuracy : {phase1_val_acc:.4f}  ({phase1_val_acc*100:.2f}%)")

# ── 1. Load the best checkpoint saved during Phase-1 ─────────────────────────
best_model_path = str(DRIVE_OUTPUT / "best_model.h5")
print(f"\n  Loading checkpoint: {best_model_path}")
model = tf.keras.models.load_model(best_model_path)

# ── 2. Locate the nested MobileNetV2 sub-model ───────────────────────────────
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise RuntimeError("❌ Could not find nested MobileNetV2 sub-model in the loaded checkpoint.")

total_layers = len(base_model.layers)
freeze_until  = total_layers - 20          # keep [0 … N-21] frozen
print(f"\n  MobileNetV2 total layers  : {total_layers}")
print(f"  Frozen up to layer index  : {freeze_until - 1}")
print(f"  Unfrozen layers           : {total_layers - freeze_until}  (last 20)")

# ── 3. Freeze early, unfreeze last 20 ────────────────────────────────────────
base_model.trainable = True                        # must be True before per-layer control
for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= freeze_until)

# Confirm counts
trainable_after  = sum(1 for l in base_model.layers if l.trainable)
frozen_after     = sum(1 for l in base_model.layers if not l.trainable)
print(f"  Layers now trainable      : {trainable_after}")
print(f"  Layers still frozen       : {frozen_after}")

# ── 4. Recompile with tiny learning rate ─────────────────────────────────────
FINE_TUNE_LR = 1e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)
print(f"\n  Recompiled — learning rate: {FINE_TUNE_LR}")

# ── 5. Fine-tuning callbacks ──────────────────────────────────────────────────
FINE_TUNE_EPOCHS  = 5
finetuned_path    = str(DRIVE_OUTPUT / "best_model_finetuned.h5")

ft_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=finetuned_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ── 6. Train Phase-2 ──────────────────────────────────────────────────────────
print(f"\n🚀 Fine-tuning for up to {FINE_TUNE_EPOCHS} epochs …")
ft_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=ft_callbacks,
)
print("✅ Fine-tuning complete")

# ── 7. Before / After comparison ──────────────────────────────────────────────
ft_h = ft_history.history
phase2_val_acc = max(ft_h["val_accuracy"])

print("\n" + "=" * 55)
print("  ACCURACY COMPARISON")
print("=" * 55)
print(f"  Phase-1 (frozen backbone)  : {phase1_val_acc*100:.2f}%")
print(f"  Phase-2 (fine-tuned)       : {phase2_val_acc*100:.2f}%")
delta = (phase2_val_acc - phase1_val_acc) * 100
sign  = "+" if delta >= 0 else ""
print(f"  Δ improvement              : {sign}{delta:.2f}%")
print("=" * 55)

# ── 8. Plot Phase-2 curves ────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ft_h["loss"],     label="Train Loss",      linewidth=2)
axes[0].plot(ft_h["val_loss"], label="Val Loss",        linewidth=2, linestyle="--")
axes[0].set_title("Fine-tune — Loss", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ft_h["accuracy"],     label="Train Accuracy", linewidth=2)
axes[1].plot(ft_h["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
axes[1].axhline(phase1_val_acc, color="grey", linestyle=":", linewidth=1.5,
                label=f"Phase-1 baseline ({phase1_val_acc*100:.1f}%)")
axes[1].set_title("Fine-tune — Accuracy", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ Fine-tuned model saved → {finetuned_path}")
print("   To use it in Grad-CAM / app.py, update MODEL_PATH to point to best_model_finetuned.h5")
\n\n# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — COMPUTE OPTIMAL DECISION THRESHOLD
# Run after fine-tuning (Cell 13) is complete
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import json
from sklearn.metrics import roc_curve, f1_score

print("Computing optimal threshold from validation set...")

# Load finetuned model
finetuned_path = str(DRIVE_OUTPUT / "best_model_finetuned.h5")
threshold_model = tf.keras.models.load_model(finetuned_path)

# Get predictions on full validation set
val_gen.reset()
val_preds = threshold_model.predict(val_gen, verbose=1)
val_labels = val_gen.classes[:len(val_preds)]

# Method 1: Best F1 Score threshold
thresholds = np.arange(0.3, 0.85, 0.01)
f1_scores  = []
for t in thresholds:
    preds = (val_preds.flatten() > t).astype(int)
    f1_scores.append(f1_score(val_labels, preds))

best_f1_threshold = thresholds[np.argmax(f1_scores)]

# Method 2: Youden's J (best for medical — balances sensitivity/specificity)
fpr, tpr, roc_thresholds = roc_curve(val_labels, val_preds)
youden_j = tpr - fpr
best_youden_threshold = float(roc_thresholds[np.argmax(youden_j)])

print("=" * 50)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 50)
print(f"Best F1 threshold       : {best_f1_threshold:.2f}")
print(f"Best Youden's J thresh  : {best_youden_threshold:.2f}")
print(f"\nRecommended (Youden's J): {best_youden_threshold:.2f}")
print("(Youden's J preferred for cancer detection —")
print(" minimizes missed cancers)")
print("=" * 50)

# Save to Drive alongside model
threshold_data = {
    'threshold': best_youden_threshold,
    'f1_threshold': float(best_f1_threshold),
    'method': "Youden's J statistic",
    'note': 'Use threshold value in app.py prediction logic'
}

threshold_path = DRIVE_OUTPUT / "model_threshold.json"
with open(threshold_path, 'w') as f:
    json.dump(threshold_data, f, indent=2)

print(f"\n✅ Threshold saved to: {threshold_path}")
print(f"   Update THRESHOLD in app.py to: {best_youden_threshold:.2f}")
\n\n# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — FINAL TRAINING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  FINAL TRAINING SUMMARY")
print("=" * 55)
print(f"\n  Phase 1 val accuracy   : {phase1_val_acc*100:.2f}%")
print(f"  Phase 2 val accuracy   : {phase2_val_acc*100:.2f}%")
delta = (phase2_val_acc - phase1_val_acc) * 100
print(f"  Improvement            : +{delta:.2f}%")
print(f"\n  Optimal threshold      : {best_youden_threshold:.2f}")
print(f"\n  Files saved to Drive:")
for f in sorted(DRIVE_OUTPUT.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"    {f.name:30s} {size_kb:>8.1f} KB")
print("\n  Next steps:")
print("  1. Copy best_model_finetuned.h5 to your app folder")
print("  2. Copy model_threshold.json to your app folder")
print(f"  3. Set THRESHOLD = {best_youden_threshold:.2f} in app.py")
print("  4. Run: streamlit run app.py")
print("=" * 55)
