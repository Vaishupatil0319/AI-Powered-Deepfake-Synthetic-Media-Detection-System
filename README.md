# AI-Powered Deepfake & Synthetic Media Detection System

> GitHub description: An AI-powered synthetic media detection prototype that analyzes images, videos and audio using machine learning, deep learning, computer vision and score-level multimodal fusion.

## 1. Overview

An educational MCA Data Science prototype for image, video, and audio analysis. It includes trainable ML and transfer-learning pipelines, sample-based video analysis, acoustic feature classification, a transparent multimodal score combiner, and a Streamlit interface. No dataset or trained artifact is bundled. Training metrics and predictions are generated only after supplying data and fitting the models.

## 2. Problem Statement

AI-generated and manipulated media can be difficult to distinguish from authentic content. This project demonstrates an extensible classification workflow while making clear that model outputs depend on the training data and do not prove authenticity.

## 3. Motivation

The project joins data handling, feature engineering, classical ML, deep learning, computer vision, audio processing, evaluation, and UI integration in a student-sized Python project.

## 4. Objectives

- Train/evaluate image baselines and a lightweight transfer-learning CNN.
- Reuse the image CNN on sampled video frames.
- Train an audio classifier on acoustic features extracted from local recordings.
- Combine available modality scores with documented score-level fusion.
- Display outputs and limitations in Streamlit without fabricated values.

## 5. Key Features

### Image Detection

The app uses either the trained MobileNetV2 classifier or Logistic Regression over 95 generic image statistics. Neither identifies a manipulation subtype.

### Video Detection

OpenCV reads video metadata and samples up to 12 evenly spaced frames by default (the app exposes 8–16). The same image CNN classifies the sampled frames; their mean synthetic score is reported. Analysis is limited to 120 seconds and 100 MB.

### Audio Detection

Librosa converts audio to mono 16 kHz and summarizes MFCCs, mel-band energy, spectral centroid/bandwidth/rolloff, zero-crossing rate, and RMS. A locally trained scaled Logistic Regression produces an uncalibrated score. Files up to 50 MB and 60 seconds are analyzed. WAV is recommended; MP3/M4A decoding may require FFmpeg.

### Multimodal Risk Analysis

The app combines only successfully analyzed modalities. Default equal weights are normalized over the available scores, so one modality passes through and two modalities are averaged by two. Risk bands are application-level thresholds: `<0.40 LOW`, `0.40–<0.70 MEDIUM`, `≥0.70 HIGH`; these are not universal forensic thresholds.

## 6. Technology Stack

Python, NumPy, Pandas (notebook analysis), scikit-learn, TensorFlow/Keras, OpenCV, Pillow, librosa, soundfile, Streamlit, Matplotlib, Seaborn, Joblib, Jupyter, Git/GitHub.

## 7. System Architecture

```text
                USER
                 │
         ┌───────┼───────┐
         ↓       ↓       ↓
       IMAGE   VIDEO    AUDIO
         ↓       ↓       ↓
      Vision   Frames   Features
       Model    Model     +
         ↓       ↓    Audio Model
       Score   Score       ↓
         └───────┼────────┘
                 ↓
         Score-Level Fusion
                 ↓
         Final Risk Score
                 ↓
           Streamlit UI
```

Data flow: image → preprocess → vision/ML model → score; video → validate/limit → sample frames → same vision model → average frame scores; audio → decode/resample → extract features → audio classifier → score; available scores → normalized score-level fusion → risk interpretation → UI. This implementation uses transparent score-level fusion rather than a single end-to-end multimodal neural network.

## 8. Machine Learning Layer

`src/preprocessing.py` extracts channel statistics and histograms, luminance percentiles, gradient summaries, and block-level brightness/variation. `src/train_ml.py` fits Logistic Regression in a `StandardScaler` pipeline and a Random Forest on a stratified file-level split. It computes accuracy, synthetic-class precision/recall/F1, and confusion matrices from held-out predictions. No winner or metric is hardcoded.

## 9. Deep Learning Layer

`src/train_vision.py` uses ImageNet-pretrained MobileNetV2 without its top classifier as a frozen feature extractor, then trains a small sigmoid classification head. RGB inputs are 224×224 by default and remain on a 0–255 scale until the embedded `mobilenet_v2.preprocess_input` layer maps them to the model's expected range. Image and video inference use the same pixel scale and read the saved model input dimensions, keeping custom `--image-size` training consistent. Augmentation is part of the model and is inactive at inference. Early stopping and best-checkpoint saving are used. MobileNetV2 was pretrained on ImageNet, not specifically on deepfake data.

## 10. Video Detection

The detector validates the extension/open state, checks duration/size/FPS/frame count, samples no more than the configured limit, preprocesses decoded RGB frames to the trained vision model's input dimensions, and aggregates actual model outputs with the mean. It rejects clips under two frames, clips over 120 seconds, or files over 100 MB. Sampling can miss a brief manipulation; there is no temporal model.

## 11. Audio Detection

The fixed feature vector concatenates temporal means and standard deviations for 20 MFCC coefficients, 64 mel bands, and five spectral/energy descriptors (178 values total). Training and inference call the same `extract_audio_features` function. Training saves a StandardScaler + Logistic Regression pipeline as `models/audio_model.pkl`; the scaler is fitted only on the training split.

## 12. Multimodal AI

`src/risk_analyzer.py` validates scores and averages available modalities with normalized weights. With equal default weights, image 0.80 and audio 0.60 yield `(0.80 + 0.60) / 2 = 0.70`. Missing modalities are omitted; no division by three occurs unless all three scores are present. This is a transparent baseline, not learned cross-modal fusion.

## 13. Dataset

See [`data/README.md`](data/README.md) for layouts, supported formats, licensing/privacy reminders, and sources. Image data goes in `data/images/{real,synthetic}/`; audio goes in `data/audio/{real,synthetic}/`. The [FaceForensics dataset repository](https://github.com/ondyari/FaceForensics) describes an access/request process for image/video research data. The [ASVspoof 2021 site](https://www.asvspoof.org/index2021.html) and [Speech Deepfake dataset release](https://zenodo.org/records/4835108) provide bona fide/spoofed speech data. Read terms and follow dataset split/citation requirements. Data is not downloaded or committed automatically because datasets can be large, private, or access-restricted.

Current scripts split files, not identities/sources/sessions. Related samples can leak across train/test; source-aware grouping should be used for serious evaluation.

## 14. Installation

Use Python 3.11 in a virtual environment. The dependency ranges target Python 3.11; Python 3.13 is not compatible with this pinned TensorFlow range. OpenCV is conservatively capped at 4.10 to preserve the NumPy 1.x environment used with this TensorFlow range; the OpenCV 4.12+ wheel line uses NumPy 2 for current Python versions. TensorFlow wheel availability also varies by OS. If installation fails on your platform, use a TensorFlow-supported Python/OS combination rather than silently substituting another framework. MP3/M4A may need FFmpeg installed on the system PATH.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 15. Training

Add your licensed data first, then run from the project root:

```bash
python -m src.train_ml
python -m src.train_vision --epochs 12 --batch-size 16 --image-size 224
python -m src.train_audio
```

Each accepts `--data-dir`; each training script accepts `--output-dir`. ML/audio need at least two examples of each class and enough total examples to populate both partitions; balanced data needs at least three per class at the default 20% test split. CNN requires six per class for train/validation/test splits. These are technical minimums, not useful recommended dataset sizes. First CNN training may download ImageNet weights. Review generated metrics before using an artifact.

## 16. Running the Application

```bash
streamlit run app/app.py
```

Use IMAGE, VIDEO, and AUDIO tabs. Each model must first be trained: image offers the CNN or ML baseline, video requires the CNN, audio requires `models/audio_model.pkl`. Successful modality scores persist in the session until “Clear analyzed results” is pressed and feed the summary.

## 17. Evaluation

Training scripts produce held-out accuracy, synthetic-class precision, recall, F1, and confusion matrices; the CNN also saves training history. Accuracy is the fraction correct and can hide minority-class failures. Precision asks how many predicted-synthetic examples were actually synthetic; recall asks how many truly synthetic examples were detected; F1 balances the two. A confusion matrix shows actual versus predicted counts. No results are included because no dataset was supplied.

## 18. Limitations

- Deepfake detection changes with new generators; models can fail on unseen sources, compression, edits, or domains.
- Performance depends on label quality, diversity, consent, and split design.
- Video uses sparse frame sampling and cannot assess temporal inconsistencies.
- Audio results depend on recording/language/speaker diversity and codec decoding.
- File-level splits can leak related content; no calibration is applied.
- Scores are uncalibrated outputs, not guaranteed probabilities, authenticity proof, or forensic evidence.
- Fusion is a weighted average baseline, not a state-of-the-art multimodal model.
- No localization, face artifact confirmation, or generator attribution is implemented.

## 19. Future Scope

Potential follow-up work includes source-grouped evaluation, temporal video models, stronger audio deepfake models, Vision Transformers, improved multimodal fusion, explainable AI, score calibration, broader datasets, and deployment optimization. These are not currently implemented.

## 20. Author

Author information was not present in the project. Add your name and institution before publishing; no personal details are assumed here.

## Interview Concepts Demonstrated

1. **Machine Learning:** fit a model from labeled examples.
2. **Classification:** assign real or synthetic class labels.
3. **Feature Engineering:** summarize image/audio content as numeric descriptors.
4. **Logistic Regression:** scaled linear probabilistic baseline.
5. **Random Forest:** ensemble of decision trees for image features.
6. **Precision:** correctness among synthetic predictions.
7. **Recall:** coverage of truly synthetic examples.
8. **F1-score:** harmonic mean of precision and recall.
9. **Confusion Matrix:** table of actual versus predicted class counts.
10. **CNN:** convolutional image feature extractor.
11. **Transfer Learning:** reuse frozen ImageNet MobileNetV2 features.
12. **Computer Vision:** decode, resize, sample, and classify visual input.
13. **MFCC:** compact representation of short-time speech spectral shape.
14. **Spectrogram:** frequency energy over time; mel spacing approximates perceptual scale.
15. **Video Frame Sampling:** classify evenly spaced frames instead of every frame.
16. **Score-Level Fusion:** combine modality outputs after independent inference.
17. **Multimodal AI:** use image, video, and audio signals in one application workflow.
18. **Overfitting:** strong training performance may not transfer to held-out sources.
19. **Data Leakage:** related sources across splits can inflate evaluation.
20. **Model Confidence vs Probability:** sigmoid output is uncalibrated and is not a guaranteed probability of truth.
