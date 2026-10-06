"""Train/evaluate a MobileNetV2 transfer-learning classifier on local images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import IMAGE_EXTENSIONS, load_image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "images",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "vision_model",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of images to use from each class.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Import TensorFlow
    # ---------------------------------------------------------
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required. Install dependencies from requirements.txt."
        ) from exc

    print("=" * 60)
    print("DEEP LEARNING / VISION MODEL TRAINING")
    print("=" * 60)

    print(f"Dataset directory : {args.data_dir}")
    print(f"Image size        : {args.image_size}x{args.image_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Maximum per class : {args.max_per_class}")
    print(f"Maximum epochs    : {args.epochs}")
    print()

    # ---------------------------------------------------------
    # Collect image paths
    # ---------------------------------------------------------
    entries: list[tuple[Path, int]] = []

    rng = np.random.default_rng(args.seed)

    for folder, label in (("real", 0), ("synthetic", 1)):

        class_dir = args.data_dir / folder

        if not class_dir.is_dir():
            raise SystemExit(
                f"Missing class directory: {class_dir}"
            )

        class_paths = [
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(
            f"Found {len(class_paths)} images in {folder}"
        )

        # -----------------------------------------------------
        # Sample BEFORE loading images
        # -----------------------------------------------------
        if args.max_per_class is not None:

            if args.max_per_class <= 0:
                raise SystemExit(
                    "--max-per-class must be greater than 0."
                )

            if len(class_paths) > args.max_per_class:

                selected_indices = rng.choice(
                    len(class_paths),
                    size=args.max_per_class,
                    replace=False,
                )

                class_paths = [
                    class_paths[index]
                    for index in selected_indices
                ]

                print(
                    f"Using {len(class_paths)} randomly selected "
                    f"{folder} images."
                )

        # -----------------------------------------------------
        # Validate selected images
        # -----------------------------------------------------
        valid_count = 0

        for path in class_paths:

            try:
                load_image(path)
                entries.append((path, label))
                valid_count += 1

            except ValueError as exc:

                print(
                    f"Skipping invalid file {path}: {exc}"
                )

        print(
            f"Valid {folder} images: {valid_count}"
        )

    # ---------------------------------------------------------
    # Check dataset
    # ---------------------------------------------------------
    if not entries:
        raise SystemExit(
            "No valid images were found."
        )

    paths = np.asarray(
        [str(path) for path, _ in entries]
    )

    labels = np.asarray(
        [label for _, label in entries],
        dtype=np.int32,
    )

    real_count = int((labels == 0).sum())
    synthetic_count = int((labels == 1).sum())

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images     : {len(paths)}")
    print(f"Real images      : {real_count}")
    print(f"Synthetic images : {synthetic_count}")

    if min(real_count, synthetic_count) < 6:
        raise SystemExit(
            "Need at least six valid images in each class."
        )

    # ---------------------------------------------------------
    # Train / validation / test split
    #
    # 70% train
    # 15% validation
    # 15% test
    # ---------------------------------------------------------
    print()
    print("Creating stratified train/validation/test split...")

    train_paths, held_paths, train_y, held_y = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=args.seed,
        stratify=labels,
    )

    val_paths, test_paths, val_y, test_y = train_test_split(
        held_paths,
        held_y,
        test_size=0.50,
        random_state=args.seed,
        stratify=held_y,
    )

    print(f"Training samples   : {len(train_paths)}")
    print(f"Validation samples : {len(val_paths)}")
    print(f"Testing samples    : {len(test_paths)}")

    # ---------------------------------------------------------
    # TensorFlow datasets
    # ---------------------------------------------------------
    size = (args.image_size, args.image_size)

    def make_dataset(
        filepaths: np.ndarray,
        targets: np.ndarray,
        training: bool,
    ):

        def load(path, label):

            raw_path = path.numpy().decode("utf-8")

            image = load_image(
                raw_path,
                size,
            )

            array = np.asarray(
                image,
                dtype=np.float32,
            )

            return array, np.float32(label)

        def wrapped(path, label):

            image, label = tf.py_function(
                load,
                [path, label],
                [tf.float32, tf.float32],
            )

            image.set_shape(
                [size[1], size[0], 3]
            )

            label.set_shape([])

            return image, label

        dataset = tf.data.Dataset.from_tensor_slices(
            (filepaths, targets)
        )

        dataset = dataset.map(
            wrapped,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if training:

            dataset = dataset.shuffle(
                len(filepaths),
                seed=args.seed,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(
            args.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset

    train_ds = make_dataset(
        train_paths,
        train_y,
        True,
    )

    val_ds = make_dataset(
        val_paths,
        val_y,
        False,
    )

    test_ds = make_dataset(
        test_paths,
        test_y,
        False,
    )

    # ---------------------------------------------------------
    # Data augmentation
    # ---------------------------------------------------------
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal"
            ),

            tf.keras.layers.RandomRotation(
                0.05
            ),

            tf.keras.layers.RandomZoom(
                0.08
            ),
        ],
        name="augmentation",
    )

    # ---------------------------------------------------------
    # MobileNetV2 Transfer Learning
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("BUILDING MOBILENETV2 MODEL")
    print("=" * 60)

    print(
        "Downloading/loading ImageNet pretrained weights "
        "if required..."
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(
            *size,
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained feature extractor
    base.trainable = False

    inputs = tf.keras.Input(
        shape=(
            *size,
            3,
        )
    )

    x = augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        x
    )

    x = base(
        x,
        training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="synthetic_probability",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    model.summary()

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------
    best_path = (
        args.output_dir / "best_model.keras"
    )

    print()
    print("Loading best model...")

    model = tf.keras.models.load_model(
        best_path
    )

    # ---------------------------------------------------------
    # Test predictions
    # ---------------------------------------------------------
    print("Evaluating on test dataset...")

    raw_predictions = model.predict(
        test_ds,
        verbose=1,
    ).reshape(-1)

    predicted = (
        raw_predictions >= 0.5
    ).astype(np.int32)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics = {

        "accuracy": float(
            accuracy_score(
                test_y,
                predicted,
            )
        ),

        "precision_synthetic": float(
            precision_score(
                test_y,
                predicted,
                zero_division=0,
            )
        ),

        "recall_synthetic": float(
            recall_score(
                test_y,
                predicted,
                zero_division=0,
            )
        ),

        "f1_synthetic": float(
            f1_score(
                test_y,
                predicted,
                zero_division=0,
            )
        ),

        "confusion_matrix_labels_real_synthetic":
            confusion_matrix(
                test_y,
                predicted,
                labels=[0, 1],
            ).tolist(),

        "test_examples": int(
            len(test_y)
        ),

        "training_examples": int(
            len(train_y)
        ),

        "validation_examples": int(
            len(val_y)
        ),

        "dataset_max_per_class":
            args.max_per_class,
    }

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------
    metrics_path = (
        args.output_dir / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    model.save(
        args.output_dir / "model.keras"
    )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------
    history_path = (
        args.output_dir / "history.json"
    )

    history_path.write_text(
        json.dumps(
            {
                key: [
                    float(value)
                    for value in values
                ]

                for key, values
                in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("VISION MODEL RESULTS")
    print("=" * 60)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print(
        f"Saved model and training history to:"
    )

    print(
        args.output_dir
    )

    print()
    print(
        "Test confusion matrix "
        "(true rows, predicted columns):"
    )

    print(
        metrics[
            "confusion_matrix_labels_real_synthetic"
        ]
    )


if __name__ == "__main__":
    main()