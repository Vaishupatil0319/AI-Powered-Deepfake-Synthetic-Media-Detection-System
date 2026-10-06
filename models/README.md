# Model artifacts

Generated model files are excluded from Git. Run `python -m src.train_ml` to produce baseline artifacts, `python -m src.train_vision` to train the MobileNetV2 model saved as `models/vision_model/model.keras`, and `python -m src.train_audio` to produce `models/audio_model.pkl`. The video detector reuses the trained vision model. The Streamlit app reports missing models rather than synthesizing predictions.
