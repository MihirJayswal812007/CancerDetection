import os
from pathlib import Path
from typing import Dict, Tuple

try:
    from model.utils import create_image_generators
except ModuleNotFoundError:
    from utils import create_image_generators

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATASET_DIR = PROJECT_ROOT / "dataset"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs"
COLAB_DRIVE_ROOT = Path("/content/drive")
COLAB_DATASET_DIR = Path("/content/drive/MyDrive/cancer-detection/dataset")
COLAB_OUTPUT_DIR = Path("/content/drive/MyDrive/cancer-detection/outputs")
IMAGE_SIZE = (224, 224)
DEFAULT_EPOCHS = 5


def is_running_in_colab() -> bool:
    """Return True when the script is executed inside Google Colab."""
    try:
        import google.colab  # type: ignore

        return True
    except ImportError:
        return False


def setup_environment() -> Tuple[Path, Path]:
    """Resolve dataset and output paths for local runs or Google Colab."""
    if is_running_in_colab():
        from google.colab import drive  # type: ignore

        if not COLAB_DRIVE_ROOT.exists():
            drive.mount("/content/drive")
        elif not (COLAB_DRIVE_ROOT / "MyDrive").exists():
            drive.mount("/content/drive", force_remount=True)

        dataset_path = COLAB_DATASET_DIR
        output_path = COLAB_OUTPUT_DIR
    else:
        dataset_path = LOCAL_DATASET_DIR
        output_path = LOCAL_OUTPUT_DIR

    return dataset_path, output_path


def load_datasets(dataset_path: Path):
    """Return the train and validation generators used for model training."""
    return create_image_generators(dataset_path=str(dataset_path))


def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
    """Build a MobileNetV2 binary classifier with a frozen ImageNet backbone."""
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cancer_detection_mobilenetv2")
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    return model


def get_parameter_counts(model: Model) -> Dict[str, int]:
    """Return trainable and non-trainable parameter counts."""
    trainable_params = int(
        sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)
    )
    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": trainable_params + non_trainable_params,
    }


def train_model(
    model: Model,
    train_generator,
    validation_generator,
    epochs: int = DEFAULT_EPOCHS,
):
    """Fit the binary classifier using the prepared generators."""
    return model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
    )


def save_training_artifacts(model: Model, history, output_path: Path) -> None:
    """Save the trained model and core training plots to the output directory."""
    os.makedirs(output_path, exist_ok=True)

    model.save(output_path / "model.h5")

    loss_plot_path = output_path / "loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    accuracy_plot_path = output_path / "accuracy.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_plot_path)
    plt.close()


if __name__ == "__main__":
    dataset_path, output_path = setup_environment()
    os.makedirs(output_path, exist_ok=True)
    train_generator, validation_generator = create_image_generators(dataset_path=str(dataset_path))
    model = build_model()
    parameter_counts = get_parameter_counts(model)
    history = train_model(model, train_generator, validation_generator)
    save_training_artifacts(model, history, output_path)

    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class labels: {train_generator.class_indices}")
    model.summary()
    print(f"Trainable parameters: {parameter_counts['trainable']:,}")
    print(f"Non-trainable parameters: {parameter_counts['non_trainable']:,}")
    print(f"Total parameters: {parameter_counts['total']:,}")
