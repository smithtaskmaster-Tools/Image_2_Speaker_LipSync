# Image 2 Speaker LipSync

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/smithtaskmastergmailcom)

Automated AI podcast video generator. Takes a studio photo with two speakers and a podcast audio file, diarizes who speaks when, lip-syncs each speaker using Wav2Lip, and composites the result back onto the original background.

**~20–25 minutes to render a 30-minute episode** on an NVIDIA GPU.

---

## What it does

1. **Diarize** — PyAnnote splits the audio into per-speaker WAV files and a speaking timeline
2. **Detect faces** — MediaPipe / Haar cascade locates both faces in the background image
3. **Animate** — Wav2Lip lip-syncs each speaker's head crop to their audio
4. **Composite** — FFmpeg overlays the animated heads back onto the original image, only when each speaker is talking

---

## Requirements

- Linux (tested on Ubuntu 24.04)
- Python 3.10+
- NVIDIA GPU with CUDA 12+ (8GB+ VRAM recommended)
- `ffmpeg` installed (`sudo apt install ffmpeg`)
- A HuggingFace account with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

---

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/babyrage-podcast-pipeline.git
cd babyrage-podcast-pipeline
```

### 2. Clone Wav2Lip

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git ~/Wav2Lip
```

### 3. Create the main Python environment

```bash
python3 -m venv ~/ai-env
~/ai-env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
~/ai-env/bin/pip install pyannote.audio huggingface_hub soundfile numpy opencv-python mediapipe pillow
```

### 4. Create the Wav2Lip environment

```bash
python3 -m venv ~/wav2lip-env

# Point wav2lip-env at ai-env's packages (avoids re-downloading torch)
echo "$HOME/ai-env/lib/python3.$(python3 -c 'import sys; print(sys.version_info.minor)')/site-packages" \
  > ~/wav2lip-env/lib/python3.$(python3 -c 'import sys; print(sys.version_info.minor)')/site-packages/ai-env-packages.pth

~/wav2lip-env/bin/pip install insightface
```

### 5. Download model weights

```bash
bash setup_wav2lip.sh
```

This downloads `wav2lip_gan.pth` (416 MB) and the s3fd face detector (86 MB).

### 6. Authenticate with HuggingFace

```bash
~/ai-env/bin/huggingface-cli login
```

Accept the pyannote model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

---

## Usage

### Place your files

```
Input/Audio/Source/  ← your podcast audio (.wav / .mp3 / .flac)
Input/Image/Source/  ← your studio background image (.png / .jpg)
```

### Run

```bash
./Py/run.sh
```

Output lands at `Py/output/DD.MM.YY/<audioname>.mp4`.

### Options

```bash
./Py/run.sh --skip-diarize   # reuse diarization from last run (saves ~5 min)
./Py/run.sh --audio /path/to/audio.wav --img /path/to/image.png --output /path/to/out.mp4
```

### Custom venv paths

```bash
PIPELINE_PYTHON=~/my-env/bin/python3 \
WAV2LIP_PYTHON=~/my-wav2lip-env/bin/python3 \
WAV2LIP_DIR=~/my-wav2lip \
./Py/run.sh
```

---

## Tips

- **Profile faces**: Wav2Lip works best on near-frontal or 3/4 angle faces. Full side profiles produce worse lip sync — adjust your studio image accordingly.
- **Image changes between episodes**: The pipeline caches face detection per image. Drop a new image and it auto-re-detects.
- **GPU power**: On laptops, ensure the GPU is running at full TDP (not battery/power-save mode) for best speed.

---

## Support

This project is free and open source. The underlying models (Wav2Lip, PyAnnote) are the work of their respective authors — all credit to them. If the pipeline saved you time, a small donation is appreciated but never required.

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/smithtaskmastergmailcom)

## License

MIT — free to use, modify, and distribute.
