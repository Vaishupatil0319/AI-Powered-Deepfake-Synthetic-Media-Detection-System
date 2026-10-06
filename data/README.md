# Dataset setup

No images are bundled with this project. Obtain data only from sources whose terms permit your use, and keep the downloaded dataset local (the `data/` contents are git-ignored).

Place images in this structure:

```text
data/images/
├── real/
│   ├── example_001.jpg
│   └── ...
└── synthetic/
    ├── example_001.png
    └── ...
```

Use `real` for authentic images and `synthetic` for AI-generated or manipulated images, with the class definition documented for the dataset you choose. Supported extensions: JPG/JPEG, PNG, WebP, BMP. Corrupt files are skipped and reported. Include varied subjects, sources, image sizes, and generation methods where licensing and project scope allow. Avoid near-duplicate images across splits; where possible split by source/person/generation campaign to reduce leakage. A binary label does not establish the specific manipulation type.

For image/video research, see the [FaceForensics dataset repository](https://github.com/ondyari/FaceForensics); dataset access and use are governed by its terms and request process. Its video-derived frames should be split by source video to avoid leakage. Review the dataset's license/access terms and intended use before downloading. This project does not fetch data automatically.

## Audio dataset setup

Keep audio recordings local and use this structure:

```text
data/audio/
├── real/
│   ├── real_001.wav
│   └── ...
└── synthetic/
    ├── synthetic_001.wav
    └── ...
```

Use authentic recordings in `real` and AI-generated/voice-converted recordings in `synthetic`. Keep the source, language, speaker-consent basis, codec, and synthesis method documented. Use appropriately licensed and consented speech; avoid placing private recordings in a public repository. WAV is recommended for consistent training. MP3, FLAC, OGG, and M4A are accepted by the inference utility; MP3/M4A decoding may require a system FFmpeg installation. Audio is converted to mono and resampled to 16 kHz, with at most 60 seconds analyzed per file.

Train the audio baseline after adding data with `python -m src.train_audio`. At least two readable clips per class and enough total clips to populate both partitions are required; balanced data needs at least three per class at the default 20% stratified held-out split. Meaningful evaluation needs a substantially larger and diverse dataset. The script extracts MFCC, mel-band, and spectral/energy summaries and evaluates a scaled Logistic Regression. Avoid speaker/source leakage: this simple script splits recordings, not speakers or recording sessions.

The [ASVspoof 2021 challenge site](https://www.asvspoof.org/index2021.html) and its [Speech Deepfake database release](https://zenodo.org/records/4835108) provide bona fide and spoofed speech data generated with TTS/voice-conversion attacks. Follow their challenge/database terms and split guidance. Convert/copy permitted samples into the two folders above; do not assume the challenge's official training/evaluation split is preserved by this project's current file-level split script.
