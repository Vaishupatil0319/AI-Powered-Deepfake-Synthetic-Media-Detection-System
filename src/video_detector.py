"""Sample video frames and classify them with the existing trained image CNN."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from src.image_detector import DEFAULT_VISION_MODEL, _load_keras_model, model_image_size
from src.preprocessing import image_to_mobilenet_input

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_DURATION_SECONDS = 120
MAX_VIDEO_FILE_BYTES = 100 * 1024 * 1024


def _materialize_video(source: str | Path | bytes | BinaryIO,
                       filename: str | None = None) -> tuple[Path, bool]:
    """Return (path, cleanup-required) for OpenCV; byte uploads use a temporary file."""
    if isinstance(source, (str, Path)):
        return Path(source), False
    suffix = Path(filename or getattr(source, "name", "upload.mp4")).suffix.lower() or ".mp4"
    if isinstance(source, bytes):
        payload = source
    elif hasattr(source, "getvalue"):
        payload = source.getvalue()
    else:
        position = source.tell()
        payload = source.read()
        source.seek(position)
    temp = tempfile.NamedTemporaryFile(prefix="synthetic_video_", suffix=suffix, delete=False)
    try:
        temp.write(payload)
        temp.close()
    except Exception:
        temp.close()
        Path(temp.name).unlink(missing_ok=True)
        raise
    return Path(temp.name), True


def predict_video(source: str | Path | bytes | BinaryIO, *, filename: str | None = None,
                  max_frames: int = 12, model_path: str | Path | None = None,
                  suspicious_threshold: float = 0.5,
                  max_duration_seconds: int = MAX_VIDEO_DURATION_SECONDS,
                  max_file_bytes: int = MAX_VIDEO_FILE_BYTES) -> dict[str, object]:
    """Analyze evenly spaced frames and average their synthetic scores.

    Requires a trained image CNN. The aggregate is a transparent frame-sampling
    baseline; it does not analyze temporal consistency or every frame.
    """
    if not 1 <= max_frames <= 64:
        raise ValueError("max_frames must be between 1 and 64")
    if max_duration_seconds <= 0 or max_file_bytes <= 0:
        raise ValueError("Video size and duration limits must be positive")
    if not 0.0 <= suspicious_threshold <= 1.0:
        raise ValueError("suspicious_threshold must be between 0 and 1")
    path, cleanup = _materialize_video(source, filename)
    try:
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format '{path.suffix}'. Use MP4, MOV, AVI, MKV, or WebM.")
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Video file could not be accessed: {exc}") from exc
        if file_size > max_file_bytes:
            raise ValueError(f"Video exceeds the {max_file_bytes // (1024 * 1024)} MB limit. Use a smaller clip.")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video analysis. Install requirements.txt.") from exc
        try:
            capture = cv2.VideoCapture(str(path))
        except cv2.error as exc:
            raise ValueError(f"Video could not be opened: {exc}") from exc
        if not capture.isOpened():
            capture.release()
            raise ValueError("Video could not be opened. Check that the file is valid and uses a supported codec.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count < 2 or not np.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
                raise ValueError("Video metadata is invalid or there are too few frames to analyze (minimum two).")
            duration_seconds = frame_count / fps
            if duration_seconds > max_duration_seconds:
                raise ValueError(f"Video is {duration_seconds:.1f} seconds long; the limit is {max_duration_seconds} seconds. Use a shorter clip.")
            frame_indices = np.unique(np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int))
            frames: list[Image.Image] = []
            actual_indices: list[int] = []
            for index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                actual_indices.append(int(index))
        except cv2.error as exc:
            raise ValueError(f"Video could not be decoded: {exc}") from exc
        finally:
            capture.release()

        if len(frames) < 2:
            raise ValueError("Fewer than two sampled frames could be decoded; the video is too short or damaged.")
        model = _load_keras_model(Path(model_path) if model_path else DEFAULT_VISION_MODEL)
        batch = np.stack([image_to_mobilenet_input(frame, model_image_size(model)) for frame in frames])
        try:
            scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"The image model could not analyze sampled video frames: {exc}") from exc
        if len(scores) != len(frames) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("The image model returned invalid frame scores.")

        frame_results = []
        for index, frame, score in zip(actual_indices, frames, scores):
            value = float(score)
            frame_results.append({"frame_index": index, "timestamp_seconds": index / fps,
                                  "synthetic_probability": value,
                                  "label": "SUSPICIOUS" if value >= suspicious_threshold else "REAL",
                                  "image": frame})
        video_score = float(np.mean(scores))
        suspicious_count = sum(item["synthetic_probability"] >= suspicious_threshold for item in frame_results)
        return {
            "label": "POSSIBLY SYNTHETIC" if video_score >= 0.5 else "LIKELY REAL",
            "synthetic_probability": video_score,
            "suspicious_frames": int(suspicious_count),
            "frames_analyzed": len(frame_results),
            "frame_results": frame_results,
            "metadata": {"duration_seconds": duration_seconds, "fps": fps, "total_frames": frame_count,
                         "sampled_frames": len(frame_results), "width": width, "height": height},
            "model": "MobileNetV2 transfer-learning CNN (sampled frames)",
        }
    finally:
        if cleanup:
            path.unlink(missing_ok=True)
