"""Train/evaluate Logistic Regression and Random Forest on image statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_feature_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:

    # ---------------------------------------------------------
    # Command-line arguments
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help=(
            "Maximum number of images to use from each class "
            "before feature extraction. Example: 5000"
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit(
            "--test-size must be between 0 and 1 (exclusive)."
        )

    if (
        args.max_per_class is not None
        and args.max_per_class <= 0
    ):
        raise SystemExit(
            "--max-per-class must be greater than 0."
        )

    # ---------------------------------------------------------
    # Load and extract dataset features
    #
    # IMPORTANT:
    # max_per_class is applied BEFORE feature extraction.
    # ---------------------------------------------------------

    print("\nLoading image dataset...")

    if args.max_per_class is not None:
        print(
            f"Using maximum {args.max_per_class} "
            "images per class."
        )
    else:
        print("Using all available images.")

    X, y, paths = load_feature_dataset(
        args.data_dir,
        max_per_class=args.max_per_class,
        random_state=42,
    )

    print(
        f"\nSuccessfully processed {len(paths)} images."
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Real images: {(y == 0).sum()}"
    )

    print(
        f"Synthetic images: {(y == 1).sum()}"
    )

    # ---------------------------------------------------------
    # Check class counts
    # ---------------------------------------------------------

    class_counts = [
        (y == label).sum()
        for label in (0, 1)
    ]

    test_count = math.ceil(
        args.test_size * len(y)
    )

    if (
        min(class_counts) < 2
        or test_count < 2
        or len(y) - test_count < 2
    ):
        raise SystemExit(
            "Need enough examples of both classes "
            "for train and test splits."
        )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    print("\nCreating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {

        "logistic_regression": Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: dict[
        str,
        dict[str, float],
    ] = {}

    # ---------------------------------------------------------
    # Train each model
    # ---------------------------------------------------------

    for name, model in models.items():

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Training: {name}"
        )

        print(
            f"{'=' * 60}"
        )

        # Train
        model.fit(
            X_train,
            y_train,
        )

        # Predict
        predicted = model.predict(
            X_test
        )

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        results[name] = {

            "accuracy": float(
                accuracy_score(
                    y_test,
                    predicted,
                )
            ),

            "precision_synthetic": float(
                precision_score(
                    y_test,
                    predicted,
                    zero_division=0,
                )
            ),

            "recall_synthetic": float(
                recall_score(
                    y_test,
                    predicted,
                    zero_division=0,
                )
            ),

            "f1_synthetic": float(
                f1_score(
                    y_test,
                    predicted,
                    zero_division=0,
                )
            ),
        }

        print(
            "\nClassification report:"
        )

        print(
            classification_report(
                y_test,
                predicted,
                target_names=[
                    "real",
                    "synthetic",
                ],
                zero_division=0,
            )
        )

        # -----------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predicted,
        )

        plt.figure(
            figsize=(6, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "real",
                "synthetic",
            ],
            yticklabels=[
                "real",
                "synthetic",
            ],
        )

        plt.title(
            f"{name} confusion matrix "
            "(held-out test set)"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.tight_layout()

        plt.savefig(
            args.output_dir
            / f"{name}_confusion_matrix.png",
            dpi=150,
        )

        plt.close()

        # -----------------------------------------------------
        # Save trained model
        # -----------------------------------------------------

        joblib.dump(
            model,
            args.output_dir
            / f"{name}.joblib",
        )

        if name == "logistic_regression":

            joblib.dump(
                model,
                args.output_dir
                / "ml_model.pkl",
            )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "HELD-OUT TEST METRICS"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    (
        args.output_dir
        / "ml_metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final information
    # ---------------------------------------------------------

    print(
        f"\nEvaluated {len(paths)} valid images."
    )

    print(
        f"Artifacts saved to: "
        f"{args.output_dir}"
    )

    print(
        "\nNo model is selected automatically; "
        "inspect class-wise metrics and "
        "generalization before use."
    )


if __name__ == "__main__":
    main()