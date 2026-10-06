"""Train/evaluate a lightweight acoustic-feature Logistic Regression model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_detector import AUDIO_EXTENSIONS, extract_audio_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "audio")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--test-size", type=float, default=0.2)

    # Limit the number of audio files used from each class.
    # Example: --max-per-class 500 = 500 real + 500 synthetic.
    parser.add_argument("--max-per-class", type=int, default=None)

    args = parser.parse_args()

    try:
        import librosa  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for audio training. "
            "Install the dependencies from requirements.txt."
        ) from exc

    rows: list[list[float]] = []
    labels: list[int] = []

    for class_name, label in (("real", 0), ("synthetic", 1)):
        directory = args.data_dir / class_name

        if not directory.is_dir():
            raise SystemExit(f"Missing audio class directory: {directory}")

        files = [
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]

        # Use only the requested number of files from each class.
        if args.max_per_class is not None:
            if args.max_per_class < 1:
                raise SystemExit("--max-per-class must be at least 1.")
            files = files[:args.max_per_class]

        print(f"{class_name}: {len(files)} audio files selected")

        for path in files:
            try:
                features, _ = extract_audio_features(path)
                rows.append(features.tolist())
                labels.append(label)
            except (ValueError, RuntimeError) as exc:
                print(f"Skipping unreadable audio {path}: {exc}")

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1 (exclusive).")

    test_count = math.ceil(args.test_size * len(labels))

    if (
        min(labels.count(0), labels.count(1)) < 2
        or test_count < 2
        or len(labels) - test_count < 2
    ):
        raise SystemExit(
            "Need enough recordings of both classes for train and test splits "
            "(three per class at the default 0.2 test size)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision_synthetic": float(
            precision_score(y_test, prediction, zero_division=0)
        ),
        "recall_synthetic": float(
            recall_score(y_test, prediction, zero_division=0)
        ),
        "f1_synthetic": float(
            f1_score(y_test, prediction, zero_division=0)
        ),
        "training_examples": len(y_train),
        "test_examples": len(y_test),
        "max_per_class": args.max_per_class,
        "confusion_matrix_labels_real_synthetic": confusion_matrix(
            y_test, prediction, labels=[0, 1]
        ).tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.output_dir / "audio_model.pkl")

    (args.output_dir / "audio_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    sns.heatmap(
        metrics["confusion_matrix_labels_real_synthetic"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "synthetic"],
        yticklabels=["real", "synthetic"],
    )

    plt.title("Audio model confusion matrix (held-out test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        args.output_dir / "audio_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    print(json.dumps(metrics, indent=2))
    print(
        f"Saved trained model to "
        f"{args.output_dir / 'audio_model.pkl'}"
    )


if __name__ == "__main__":
    main()