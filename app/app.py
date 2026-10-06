from pathlib import Path
import sys
import tempfile

import streamlit as st


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------

from src.image_detector import predict_image
from src.video_detector import predict_video
from src.audio_detector import predict_audio
from src.risk_analyzer import combine_scores


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Synthetic Media Analyzer",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# HELPER: SAVE UPLOADED IMAGE
# ---------------------------------------------------------

def save_uploaded_file(uploaded_file):
    """
    Save a Streamlit uploaded file temporarily
    and return the temporary file path.
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())

        return temp_file.name


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🔍 SYNTHETIC MEDIA ANALYZER")

st.caption(
    "AI-powered analysis of images, videos, and audio "
    "for synthetic-media indicators."
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Analysis Mode")

mode = st.sidebar.radio(
    "Choose media type",
    [
        "Image",
        "Video",
        "Audio",
        "Multimodal",
    ],
)


# ---------------------------------------------------------
# HELPER: DISPLAY RESULT
# ---------------------------------------------------------

def show_result(title, result):
    """Display a detector result."""

    score = float(result["synthetic_probability"])

    percentage = score * 100

    st.subheader(title)

    col1, col2, col3 = st.columns(3)

    # Synthetic probability
    with col1:

        st.metric(
            "Synthetic Media Score",
            f"{percentage:.2f}%",
        )

    # Classification
    with col2:

        display_label = (
            "POSSIBLY SYNTHETIC"
            if score >= 0.50
            else "LIKELY REAL"
        )      

        st.metric(
            "Model Assessment",
            display_label,
        )
    # Risk level
    with col3:

        if score < 0.40:
            risk = "LOW"

        elif score < 0.70:
            risk = "MEDIUM"

        else:
            risk = "HIGH"

        st.metric(
            "Risk Level",
            risk,
        )

    st.caption(
        f"Model: {result.get('model', 'Unknown')}"
    )


# =========================================================
# IMAGE MODE
# =========================================================

if mode == "Image":

    st.header("🖼️ Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button(
            "🔍 Analyze Image",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Analyzing image..."
                ):

                    # Save uploaded image temporarily
                    image_path = save_uploaded_file(
                        uploaded_file
                    )

                    # Send file path to image detector
                    result = predict_image(
                        image_path
                    )

                show_result(
                    "Image Analysis Result",
                    result,
                )

            except Exception as exc:

                st.error(
                    f"Image analysis failed: {exc}"
                )


# =========================================================
# VIDEO MODE
# =========================================================

elif mode == "Video":

    st.header("🎥 Video Analysis")

    uploaded_file = st.file_uploader(
        "Upload a short video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        if st.button(
            "🔍 Analyze Video",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Analyzing video frames..."
                ):

                    result = predict_video(
                        uploaded_file.getvalue()
                    )

                show_result(
                    "Video Analysis Result",
                    result,
                )

                st.write(
                    f"**Frames analyzed:** "
                    f"{result['frames_analyzed']}"
                )

                st.write(
                    f"**Suspicious frames:** "
                    f"{result['suspicious_frames']}"
                )

                st.json(
                    result["metadata"]
                )

            except Exception as exc:

                st.error(
                    f"Video analysis failed: {exc}"
                )


# =========================================================
# AUDIO MODE
# =========================================================

elif mode == "Audio":

    st.header("🎵 Audio Analysis")

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        if st.button(
            "🔍 Analyze Audio",
            type="primary",
        ):

            try:

                with st.spinner(
                    "Analyzing acoustic features..."
                ):

                    # Audio detector accepts uploaded bytes
                    result = predict_audio(
                        uploaded_file.getvalue()
                    )

                show_result(
                    "Audio Analysis Result",
                    result,
                )

                st.json(
                    result["metadata"]
                )

            except Exception as exc:

                st.error(
                    f"Audio analysis failed: {exc}"
                )


# =========================================================
# MULTIMODAL MODE
# =========================================================

elif mode == "Multimodal":

    st.header("🧠 Multimodal Analysis")

    st.write(
        "Upload one or more media types. "
        "Available modality scores are combined dynamically."
    )

    # -----------------------------------------------------
    # IMAGE UPLOAD
    # -----------------------------------------------------

    image_file = st.file_uploader(
        "Upload Image (optional)",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="multi_image",
    )

    # -----------------------------------------------------
    # VIDEO UPLOAD
    # -----------------------------------------------------

    video_file = st.file_uploader(
        "Upload Video (optional)",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
        key="multi_video",
    )

    # -----------------------------------------------------
    # AUDIO UPLOAD
    # -----------------------------------------------------

    audio_file = st.file_uploader(
        "Upload Audio (optional)",
        type=[
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
        key="multi_audio",
    )

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Multimodal Media",
        type="primary",
    ):

        # Check whether anything was uploaded

        if (
            image_file is None
            and video_file is None
            and audio_file is None
        ):

            st.warning(
                "Please upload at least one media file."
            )

        else:

            # Initial scores

            image_score = None
            video_score = None
            audio_score = None

            try:

                # =================================================
                # IMAGE ANALYSIS
                # =================================================

                if image_file is not None:

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        # Image detector needs a file path
                        image_path = save_uploaded_file(
                            image_file
                        )

                        image_result = predict_image(
                            image_path
                        )

                    image_score = image_result[
                        "synthetic_probability"
                    ]

                # =================================================
                # VIDEO ANALYSIS
                # =================================================

                if video_file is not None:

                    with st.spinner(
                        "Analyzing video..."
                    ):

                        video_result = predict_video(
                            video_file.getvalue()
                        )

                    video_score = video_result[
                        "synthetic_probability"
                    ]

                # =================================================
                # AUDIO ANALYSIS
                # =================================================

                if audio_file is not None:

                    with st.spinner(
                        "Analyzing audio..."
                    ):

                        audio_result = predict_audio(
                            audio_file.getvalue()
                        )

                    audio_score = audio_result[
                        "synthetic_probability"
                    ]

                # =================================================
                # MULTIMODAL FUSION
                # =================================================

                final_result = combine_scores(
                    image_score=image_score,
                    video_score=video_score,
                    audio_score=audio_score,
                )

                st.divider()

                # =================================================
                # MODALITY RESULTS
                # =================================================

                st.subheader(
                    "📊 Modality Results"
                )

                modality_count = len(
                    final_result[
                        "modality_scores"
                    ]
                )

                columns = st.columns(
                    modality_count
                )

                for column, (
                    name,
                    score,
                ) in zip(
                    columns,
                    final_result[
                        "modality_scores"
                    ].items(),
                ):

                    with column:

                        st.metric(
                            name.capitalize(),
                            f"{score * 100:.2f}%",
                        )

                st.divider()

                # =================================================
                # OVERALL RESULT
                # =================================================

                st.subheader(
                    "🚨 Overall Synthetic Media Risk"
                )

                overall = final_result[
                    "overall_synthetic_percentage"
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Overall Score",
                        f"{overall:.2f}%",
                    )

                with col2:

                    st.metric(
                        "Risk Level",
                        final_result[
                            "risk_level"
                        ],
                    )

                # Classification

                if final_result["label"] == "SYNTHETIC":

                    st.error(
                        "⚠️ Classification: SYNTHETIC"
                    )

                else:

                    st.success(
                        "✅ Classification: REAL"
                    )

                # Explanation

                st.info(
                    "The result is an AI-based screening "
                    "score and should not be treated as "
                    "forensic proof."
                )

            except Exception as exc:

                st.error(
                    f"Multimodal analysis failed: {exc}"
                )