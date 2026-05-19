#!/bin/bash
# Download Wav2Lip model weights. Run once before first pipeline use.
set -e

WAV2LIP_DIR="$HOME/Wav2Lip"
CHECKPOINTS="$WAV2LIP_DIR/checkpoints"
mkdir -p "$CHECKPOINTS"

echo "=== Downloading Wav2Lip GAN checkpoint ==="
# wav2lip_gan.pth — better visual quality than wav2lip.pth
# Hosted on Hugging Face as a reliable mirror of the original release
if [ ! -f "$CHECKPOINTS/wav2lip_gan.pth" ]; then
    wget -q --show-progress \
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth" \
        -O "$CHECKPOINTS/wav2lip_gan.pth"
    echo "  wav2lip_gan.pth downloaded ($(du -sh "$CHECKPOINTS/wav2lip_gan.pth" | cut -f1))"
else
    echo "  wav2lip_gan.pth already present — skipping"
fi

echo ""
echo "=== Downloading s3fd face detector ==="
# s3fd.pth is downloaded automatically by Wav2Lip on first inference run.
# Pre-downloading it here avoids a delay mid-pipeline.
S3FD_DIR="$WAV2LIP_DIR/face_detection/detection/sfd"
if [ ! -f "$S3FD_DIR/s3fd.pth" ]; then
    wget -q --show-progress \
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
        -O "$S3FD_DIR/s3fd.pth"
    echo "  s3fd.pth downloaded ($(du -sh "$S3FD_DIR/s3fd.pth" | cut -f1))"
else
    echo "  s3fd.pth already present — skipping"
fi

echo ""
echo "=== Verifying Wav2Lip imports ==="
cd "$WAV2LIP_DIR"
/home/ezio/wav2lip-env/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from models import Wav2Lip
import face_detection
import audio
print('  models, face_detection, audio import OK')
"

echo ""
echo "=== Setup complete ==="
echo "Run the pipeline:"
echo "  cd /home/ezio/Desktop/AI_Podcasts/Wav2Lip/Py"
echo "  ./run.sh --audio /path/to/audio.wav --img /path/to/DuoBabies.png"
