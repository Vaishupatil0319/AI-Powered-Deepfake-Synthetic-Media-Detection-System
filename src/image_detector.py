"""Load and run trained image models without inventing predictions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image

from src.preprocessing import IMAGE_SIZE, image_to_mobilenet_input, load_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISION_MODEL = PROJECT_ROOT / "models" / "vision_model" / "model.keras"
DEFAULT_ML_MODEL = PROJECT_ROOT / "models" / "ml_model.pkl"


class ModelUnavailableError(RuntimeError):
    """Raised when no trained model artifact is available."""


@lru_cache(maxsize=2)
def _load_keras_model_cached(path_string: str, modified_ns: int, file_size: int) -> Any:
    """Cache an immutable loaded model by artifact identity, not prediction result."""
    del modified_ns, file_size
    path = Path(path_string)
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except ImportError as exc:
        raise ModelUnavailableError("TensorFlow is required to load the vision model. Install requirements.txt.") from exc
    except Exception as exc:
        raise ModelUnavailableError(f"Could not load the vision model: {exc}") from exc


def _load_keras_model(path: Path) -> Any:
    """Load the saved model once, refreshing the cache if the artifact changes."""
    if not path.is_file():
        raise ModelUnavailableError(f"Trained vision model not found at {path}. Train it with src/train_vision.py.")
    resolved = path.resolve()
    stat = resolved.stat()
    return _load_keras_model_cached(str(resolved), stat.st_mtime_ns, stat.st_size)


def model_image_size(model: Any) -> tuple[int, int]:
    """Return (width, height) from the model input shape or the project default."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape and len(shape) == 4 and shape[1] and shape[2]:
        return int(shape[2]), int(shape[1])
    return IMAGE_SIZE


def predict_image(image: Image.Image | str | Path | BinaryIO, model_path: str | Path | None = None,
                  model_type: str = "vision") -> dict[str, object]:
    """Predict real/synthetic using a trained vision CNN or trained ML baseline.

    The returned score is the model's uncalibrated synthetic probability, not a
    guaranteed probability of truth. Labels are real=0, synthetic=1.
    """
    clean_image = load_image(image)
    if model_type == "vision":
        path = Path(model_path) if model_path else DEFAULT_VISION_MODEL
        model = _load_keras_model(path)
        batch = np.expand_dims(image_to_mobilenet_input(clean_image, model_image_size(model)), axis=0)
        try:
            probability = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
        except Exception as exc:
            raise RuntimeError(f"Vision model prediction failed: {exc}") from exc
        model_name = "MobileNetV2 transfer-learning CNN"
    elif model_type == "ml":
        path = Path(model_path) if model_path else DEFAULT_ML_MODEL
        if not path.is_file():
            raise ModelUnavailableError(f"Trained ML model not found at {path}. Train it with src/train_ml.py.")
        try:
            import joblib
            from src.preprocessing import extract_image_features
            model = joblib.load(path)
            probability = float(model.predict_proba([extract_image_features(clean_image)])[0, 1])
        except Exception as exc:
            raise RuntimeError(f"ML model prediction failed: {exc}") from exc
        model_name = "Hand-engineered feature baseline"
    else:
        raise ValueError("model_type must be 'vision' or 'ml'")

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("Model returned an invalid probability")
    label = "AI-GENERATED / SYNTHETIC" if probability >= 0.5 else "REAL"
    confidence = probability if probability >= 0.5 else 1.0 - probability
    return {"label": label, "synthetic_probability": probability, "confidence": confidence,
            "model": model_name, "input_type": "Image"}
