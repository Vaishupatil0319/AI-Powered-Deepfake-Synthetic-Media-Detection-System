"""Multimodal synthetic-media risk analysis."""

from __future__ import annotations


def _risk_level(score: float) -> str:
    """Map synthetic probability to a simple risk level."""
    if score < 0.40:
        return "LOW"
    if score < 0.70:
        return "MEDIUM"
    return "HIGH"


def _label(score: float) -> str:
    """Convert probability into a user-facing label."""
    return "SYNTHETIC" if score >= 0.50 else "REAL"


def combine_scores(
    image_score: float | None = None,
    video_score: float | None = None,
    audio_score: float | None = None,
) -> dict:
    """
    Combine all available modality scores.

    Only uploaded/available modalities participate in the average.
    """

    scores = {}

    if image_score is not None:
        scores["image"] = float(image_score)

    if video_score is not None:
        scores["video"] = float(video_score)

    if audio_score is not None:
        scores["audio"] = float(audio_score)

    if not scores:
        raise ValueError("At least one modality score is required.")

    overall_score = sum(scores.values()) / len(scores)

    return {
        "available_modalities": list(scores.keys()),
        "modality_scores": scores,
        "overall_synthetic_probability": overall_score,
        "overall_synthetic_percentage": overall_score * 100,
        "label": _label(overall_score),
        "risk_level": _risk_level(overall_score),
    }