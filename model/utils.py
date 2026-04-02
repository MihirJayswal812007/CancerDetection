from pathlib import Path
from typing import Tuple

from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_NAMES = ("cancer", "non-cancer")


def create_image_generators(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
):
    """Build train and validation generators for a binary cancer dataset."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    missing_classes = [
        class_name for class_name in CLASS_NAMES if not (dataset_path / class_name).exists()
    ]
    if missing_classes:
        raise ValueError(
            "Dataset directory must contain 'cancer' and 'non-cancer' subfolders. "
            f"Missing: {', '.join(missing_classes)}"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(dataset_path),
        target_size=image_size,
        batch_size=batch_size,
        classes=list(CLASS_NAMES),
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=str(dataset_path),
        target_size=image_size,
        batch_size=batch_size,
        classes=list(CLASS_NAMES),
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    return train_generator, validation_generator
