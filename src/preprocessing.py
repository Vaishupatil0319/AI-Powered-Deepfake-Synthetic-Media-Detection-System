"""Image validation, preprocessing, and hand-engineered feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_SIZE = (224, 224)


def load_image(
    source: str | Path | Image.Image | BinaryIO,
    image_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Open an image safely, apply EXIF orientation, convert to RGB and optionally resize."""
    try:
        image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")

        if image_size:
            image = image.resize(image_size, Image.Resampling.BILINEAR)

        return image

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Could not read a valid image: {exc}") from exc


def image_to_array(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize an image and return float32 RGB pixels scaled to [0, 1]."""
    return np.asarray(
        load_image(image, image_size),
        dtype=np.float32,
    ) / 255.0


def image_to_mobilenet_input(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Return RGB float32 pixels in [0, 255]."""
    return np.asarray(
        load_image(image, image_size),
        dtype=np.float32,
    )


def extract_image_features(
    image: Image.Image | str | Path | BinaryIO,
) -> np.ndarray:
    """Return deterministic visual statistics for the ML baseline."""

    rgb = np.asarray(
        load_image(image),
        dtype=np.float32,
    ) / 255.0

    gray = np.asarray(
        load_image(image, (128, 128)).convert("L"),
        dtype=np.float32,
    ) / 255.0

    features: list[float] = []

    # Color statistics
    for channel in np.moveaxis(rgb, -1, 0):
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
            ]
        )

        features.extend(
            np.histogram(
                channel,
                bins=16,
                range=(0.0, 1.0),
                density=True,
            )[0].tolist()
        )

    # Grayscale statistics
    features.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(np.percentile(gray, 10)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 90)),
        ]
    )

    # Edge statistics
    gx = np.diff(
        gray,
        axis=1,
        prepend=gray[:, :1],
    )

    gy = np.diff(
        gray,
        axis=0,
        prepend=gray[:1, :],
    )

    magnitude = np.hypot(gx, gy)

    features.extend(
        [
            float(magnitude.mean()),
            float(magnitude.std()),
            float(np.mean(magnitude > 0.1)),
            float(np.mean(magnitude > 0.25)),
        ]
    )

    # Coarse texture descriptor
    h, w = gray.shape

    for row in np.array_split(gray, 4, axis=0):
        for block in np.array_split(row, 4, axis=1):
            features.extend(
                [
                    float(block.mean()),
                    float(block.std()),
                ]
            )

    return np.asarray(features, dtype=np.float32)


def iter_dataset_images(
    root: str | Path,
) -> Iterable[tuple[Path, int]]:
    """Yield (path, label) for real=0 and synthetic=1."""

    root = Path(root)

    for class_name, label in (
        ("real", 0),
        ("synthetic", 1),
    ):
        class_dir = root / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for path in sorted(class_dir.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                yield path, label


def load_feature_dataset(
    root: str | Path,
    max_per_class: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """
    Extract image features and labels.

    If max_per_class is provided, only that many images are selected
    from each class BEFORE feature extraction. This prevents the
    program from processing the entire dataset unnecessarily.
    """

    import warnings

    # ---------------------------------------------------------
    # Step 1: collect image paths
    # ---------------------------------------------------------

    dataset_items = list(iter_dataset_images(root))

    if not dataset_items:
        raise ValueError(
            f"No readable image files found under {root}"
        )

    # ---------------------------------------------------------
    # Step 2: optionally limit images BEFORE feature extraction
    # ---------------------------------------------------------

    if max_per_class is not None:

        if max_per_class <= 0:
            raise ValueError(
                "max_per_class must be greater than 0."
            )

        rng = np.random.default_rng(random_state)

        selected_items: list[tuple[Path, int]] = []

        for label in (0, 1):

            class_items = [
                item
                for item in dataset_items
                if item[1] == label
            ]

            if not class_items:
                raise ValueError(
                    f"No images found for class label {label}."
                )

            limit = min(
                max_per_class,
                len(class_items),
            )

            selected_indices = rng.choice(
                len(class_items),
                size=limit,
                replace=False,
            )

            selected_items.extend(
                class_items[index]
                for index in selected_indices
            )

        dataset_items = selected_items

    # ---------------------------------------------------------
    # Step 3: extract features
    # ---------------------------------------------------------

    rows: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for path, label in dataset_items:

        try:
            rows.append(
                extract_image_features(path)
            )

            labels.append(label)
            paths.append(path)

        except ValueError as exc:

            warnings.warn(
                f"Skipping {path}: {exc}",
                stacklevel=2,
            )

    if not rows:
        raise ValueError(
            f"No readable images found under {root}"
        )

    return (
        np.vstack(rows),
        np.asarray(labels, dtype=np.int64),
        paths,
    )