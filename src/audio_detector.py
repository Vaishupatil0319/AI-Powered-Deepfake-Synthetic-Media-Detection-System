"""Acoustic feature extraction and inference with a locally trained classifier."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_MODEL = PROJECT_ROOT / "models" / "audio_model.pkl"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 60
MAX_AUDIO_FILE_BYTES = 50 * 1024 * 1024


def _materialize_audio(source: str | Path | bytes | BinaryIO,
                       filename: str | None = None) -> tuple[Path, bool]:
    """Return (path, cleanup-required); file-like uploads are copied to a temporary file."""
    if isinstance(source, (str, Path)):
        return Path(source), False
    suffix = Path(filename or getattr(source, "name", "upload.wav")).suffix.lower() or ".wav"
    if isinstance(source, bytes):
        payload = source
    elif hasattr(source, "getvalue"):
        payload = source.getvalue()
    else:
        position = source.tell()
        payload = source.read()
        source.seek(position)
    temp = tempfile.NamedTemporaryFile(prefix="synthetic_audio_", suffix=suffix, delete=False)
    try:
        temp.write(payload)
        temp.close()
    except Exception:
        temp.close()
        Path(temp.name).unlink(missing_ok=True)
        raise
    return Path(temp.name), True


def extract_audio_features(source: str | Path | bytes | BinaryIO, *, filename: str | None = None,
                           sample_rate: int = TARGET_SAMPLE_RATE,
                           max_duration: int = MAX_AUDIO_SECONDS,
                           max_file_bytes: int = MAX_AUDIO_FILE_BYTES) -> tuple[np.ndarray, dict[str, float | int]]:
    """Load mono audio and aggregate MFCC, mel-band, spectral, ZCR, and RMS features.

    Mean and standard deviation across time yield a fixed-size vector. WAV decoding
    uses soundfile through librosa. MP3 works when librosa's audioread backend and
    a supported system decoder (commonly FFmpeg) are installed.
    """
    if sample_rate < 1 or max_duration < 1 or max_file_bytes < 1:
        raise ValueError("sample_rate, max_duration, and max_file_bytes must be positive")
    path, cleanup = _materialize_audio(source, filename)
    try:
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format '{path.suffix}'. Use WAV or MP3 (also FLAC, OGG, M4A).")
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Audio file could not be accessed: {exc}") from exc
        if file_size > max_file_bytes:
            raise ValueError(f"Audio exceeds the {max_file_bytes // (1024 * 1024)} MB limit.")
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("librosa is required for audio analysis. Install requirements.txt.") from exc
        try:
            waveform, actual_rate = librosa.load(path, sr=sample_rate, mono=True, duration=max_duration)
        except Exception as exc:
            raise ValueError(f"Audio could not be decoded. For MP3, install a system FFmpeg decoder if required: {exc}") from exc
        if waveform.size < 2048 or not np.isfinite(waveform).all():
            raise ValueError("Audio is empty, too short, or contains invalid samples (minimum about 0.13 seconds).")
        try:
            n_fft = min(1024, 2 ** int(np.floor(np.log2(waveform.size))))
            hop = max(128, n_fft // 4)
            mfcc = librosa.feature.mfcc(y=waveform, sr=actual_rate, n_mfcc=20, n_fft=n_fft,
                                        hop_length=hop, n_mels=64)
            mel_power = librosa.feature.melspectrogram(y=waveform, sr=actual_rate, n_fft=n_fft,
                                                        hop_length=hop, n_mels=64, power=2.0)
            mel_db = librosa.power_to_db(mel_power, ref=np.max)
            descriptors = [mfcc, mel_db]
            descriptors.extend([
                librosa.feature.spectral_centroid(y=waveform, sr=actual_rate, n_fft=n_fft, hop_length=hop),
                librosa.feature.spectral_bandwidth(y=waveform, sr=actual_rate, n_fft=n_fft, hop_length=hop),
                librosa.feature.spectral_rolloff(y=waveform, sr=actual_rate, n_fft=n_fft, hop_length=hop),
                librosa.feature.zero_crossing_rate(y=waveform, frame_length=n_fft, hop_length=hop),
                librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop),
            ])
            features = np.concatenate([np.mean(values, axis=1) for values in descriptors] +
                                      [np.std(values, axis=1) for values in descriptors]).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Audio feature extraction failed: {exc}") from exc
        if not np.isfinite(features).all():
            raise ValueError("Audio feature extraction produced non-finite values.")
        metadata = {"duration_seconds": float(waveform.size / actual_rate),
                    "sample_rate": int(actual_rate), "samples": int(waveform.size)}
        return features, metadata
    finally:
        if cleanup:
            path.unlink(missing_ok=True)


def predict_audio(source: str | Path | bytes | BinaryIO, *, filename: str | None = None,
                  model_path: str | Path | None = None) -> dict[str, object]:
    """Predict real/synthetic audio using the classifier trained by `src.train_audio`."""
    features, metadata = extract_audio_features(source, filename=filename)
    path = Path(model_path) if model_path else DEFAULT_AUDIO_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"Trained audio model not found at {path}. Train it with python -m src.train_audio.")
    try:
        import joblib
        model = joblib.load(path)
        probability = float(model.predict_proba([features])[0, 1])
    except Exception as exc:
        raise RuntimeError(f"Audio model prediction failed: {exc}") from exc
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("Audio model returned an invalid synthetic score.")
    return {"label": "SYNTHETIC AUDIO" if probability >= 0.5 else "REAL AUDIO",
            "synthetic_probability": probability,
            "confidence": max(probability, 1.0 - probability),
            "metadata": metadata,
            "features": "MFCC + Mel spectrogram + spectral centroid/bandwidth/rolloff + zero-crossing rate + RMS",
            "model": "Acoustic feature Logistic Regression"}
