#!/bin/bash
# BabyRage Podcast Pipeline — Wav2Lip Edition
# Usage: ./run.sh [--skip-diarize] [--audio file] [--img file] [--output file]
#
# Override defaults via environment variables:
#   WAV2LIP_DIR    path to Wav2Lip repo     (default: ~/Wav2Lip)
#   WAV2LIP_PYTHON path to wav2lip venv python (default: ~/wav2lip-env/bin/python3)
#   PIPELINE_PYTHON path to main venv python   (default: ~/ai-env/bin/python3)

cd "$(dirname "$0")"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-$HOME/ai-env/bin/python3}"

time "$PIPELINE_PYTHON" run_podcast.py "$@"
