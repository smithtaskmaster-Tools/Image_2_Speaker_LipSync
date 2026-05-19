#!/usr/bin/env python3
"""
BabyRage Podcast Pipeline — Wav2Lip Edition
Usage: python run_podcast.py --audio input.wav --img DuoBabies.png --output episodes/ep01.mp4

Compositing strategy:
  - Background image is kept at all times (preserves branding/studio)
  - Wav2Lip animates tight head-only crops of each speaker
  - Animated head is overlaid at the exact original face position when that speaker talks
  - No side-by-side split, no black borders, no zoom jumps
  - 20px feathered edges soften the boundary between animated and static regions

NOTE: Speakers are animated sequentially because Wav2Lip writes to a hardcoded
temp/result.avi path — concurrent runs would conflict on a single GPU anyway.
"""

import argparse, subprocess, json, os
from pathlib import Path
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline as DiarizePipeline
from huggingface_hub import get_token, login as hf_login
import torch

# ── CONFIG — override via environment variables if needed ─────────────────────
HF_TOKEN        = get_token()
_HERE           = Path(__file__).parent.resolve()
WAV2LIP_DIR     = Path(os.environ.get("WAV2LIP_DIR",     str(Path.home() / "Wav2Lip")))
WAV2LIP_PYTHON  = Path(os.environ.get("WAV2LIP_PYTHON",  str(Path.home() / "wav2lip-env/bin/python3")))
WORK_DIR        = _HERE / "work"

IMG_SIZE = 512   # head crop resolution fed to Wav2Lip
FPS      = 25    # Wav2Lip native output FPS
# ─────────────────────────────────────────────────────────────────────────────


def _hf_auth():
    if HF_TOKEN:
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
    else:
        print("WARNING: HF_TOKEN not set — pyannote model download will fail if not cached.")


def diarize(audio_path: Path, work_dir: Path) -> dict:
    print("Diarizing audio...")
    _hf_auth()
    pipeline = DiarizePipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    result      = pipeline(str(audio_path))
    diarization = result.speaker_diarization

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n_samples = len(audio)

    speakers = sorted(set(s for _, _, s in diarization.itertracks(yield_label=True)))
    if len(speakers) < 2:
        raise ValueError(f"Only {len(speakers)} speaker detected — check audio quality.")
    if len(speakers) > 2:
        print(f"WARNING: {len(speakers)} speakers detected, using first 2: {speakers[:2]}")
        speakers = speakers[:2]

    speaker_map = {speakers[0]: 0, speakers[1]: 1}
    print(f"  Speaker 1 -> {speakers[0]}, Speaker 2 -> {speakers[1]}")

    tracks   = [np.zeros(n_samples, dtype=np.float32) for _ in range(2)]
    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            continue
        idx   = speaker_map[speaker]
        start = int(turn.start * sr)
        end   = min(int(turn.end * sr), n_samples)
        tracks[idx][start:end] = audio[start:end]
        timeline.append({"speaker": idx, "start": turn.start, "end": turn.end})

    paths = []
    for i, track in enumerate(tracks):
        p = work_dir / f"speaker{i+1}.wav"
        sf.write(str(p), track, sr)
        paths.append(p)
        active_s = np.count_nonzero(track) / sr
        print(f"  speaker{i+1}.wav written — active speech: {active_s:.1f}s")

    (work_dir / "timeline.json").write_text(json.dumps(timeline, indent=2))

    pipeline.to(torch.device("cpu"))
    del pipeline
    torch.cuda.empty_cache()
    print("  PyAnnote model evicted from GPU VRAM")

    return {"speaker_wavs": paths, "sr": sr, "duration": n_samples / sr, "timeline": timeline}


def _face_to_head_crop(x1, y1, x2, y2, img_w, img_h):
    """
    Tight square head crop — just head and chin, minimal background padding.
    Returns (nx1, ny1, nx2, ny2) in original image coordinates.
    """
    fw, fh = x2 - x1, y2 - y1
    cx_f   = (x1 + x2) / 2

    top    = int(y1 - 0.35 * fh)
    bottom = int(y2 + 0.35 * fh)
    left   = int(cx_f - 0.55 * fw)
    right  = int(cx_f + 0.55 * fw)

    left   = max(0, left);     top    = max(0, top)
    right  = min(img_w, right); bottom = min(img_h, bottom)

    w, h  = right - left, bottom - top
    side  = max(w, h)
    cx_c  = (left + right) // 2
    cy_c  = (top  + bottom) // 2

    nx1 = max(0,     cx_c - side // 2)
    ny1 = max(0,     cy_c - side // 2)
    nx2 = min(img_w, nx1 + side)
    ny2 = min(img_h, ny1 + side)
    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    return nx1, ny1, nx2, ny2


def extract_face_crops(img_path: Path, work_dir: Path, img_size: int = IMG_SIZE) -> tuple:
    """
    Detect the two faces in img_path, extract tight square head crops for Wav2Lip.

    Returns:
        (img1_path, img2_path)  — img_size×img_size PNGs to feed Wav2Lip
        face_boxes              — [(x1,y1,x2,y2), (x1,y1,x2,y2)] in ORIGINAL image coords
                                  used by composite() to position the animated overlay

    Detection chain:
      1. MediaPipe  — good for frontal faces
      2. OpenCV Haar profile cascade — handles side-facing / 3/4 profile faces
      3. Estimated fallback — uses left/right quarter of image
    """
    from PIL import Image
    import cv2
    _lanczos = getattr(Image, "Resampling", Image).LANCZOS

    img_pil  = Image.open(img_path).convert("RGB")
    img_w, img_h = img_pil.size
    img_rgb  = np.array(img_pil)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    raw_boxes = None

    # ── 1. MediaPipe ──────────────────────────────────────────────────────────
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        for model_sel, conf in [(0, 0.2), (1, 0.2), (0, 0.1), (1, 0.1)]:
            with mp_face.FaceDetection(model_selection=model_sel,
                                       min_detection_confidence=conf) as det:
                res = det.process(img_rgb)
            if res.detections and len(res.detections) >= 2:
                raw = []
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    raw.append((int(bb.xmin * img_w), int(bb.ymin * img_h),
                                int((bb.xmin + bb.width) * img_w),
                                int((bb.ymin + bb.height) * img_h)))
                raw_boxes = sorted(raw, key=lambda b: b[0])[:2]
                print(f"  MediaPipe: {len(res.detections)} face(s), model={model_sel} conf={conf}")
                break
    except ImportError:
        pass

    # ── 2. Haar profile cascade ───────────────────────────────────────────────
    if raw_boxes is None:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")
        min_px = int(img_w * 0.05)
        min_y_center = img_h * 0.28

        all_faces = []
        for (x, y, w, h) in cascade.detectMultiScale(img_gray, 1.05, 3,
                                                      minSize=(min_px, min_px)):
            if (y + h / 2) > min_y_center:
                all_faces.append((x, y, x + w, y + h))
        img_flip = cv2.flip(img_gray, 1)
        for (x, y, w, h) in cascade.detectMultiScale(img_flip, 1.05, 3,
                                                      minSize=(min_px, min_px)):
            if (y + h / 2) > min_y_center:
                all_faces.append((img_w - (x + w), y, img_w - x, y + h))

        def _largest(lst):
            return max(lst, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        left_h  = [b for b in all_faces if (b[0]+b[2])//2 <  img_w//2]
        right_h = [b for b in all_faces if (b[0]+b[2])//2 >= img_w//2]

        if left_h and right_h:
            raw_boxes = [_largest(left_h), _largest(right_h)]
            print(f"  Haar: {len(all_faces)} valid candidate(s), picked largest per half")
        elif right_h:
            rx1, ry1, rx2, ry2 = _largest(right_h)
            rw = rx2 - rx1
            raw_boxes = [(max(0, img_w - rx2), ry1, img_w - rx1, ry2),
                         (rx1, ry1, rx2, ry2)]
            print(f"  Haar: only right baby found — mirroring x for left")
        elif left_h:
            lx1, ly1, lx2, ly2 = _largest(left_h)
            raw_boxes = [(lx1, ly1, lx2, ly2),
                         (max(0, img_w - lx2), ly1, min(img_w, img_w - lx1), ly2)]
            print(f"  Haar: only left baby found — mirroring x for right")

    # ── 3. Estimated fallback ─────────────────────────────────────────────────
    if raw_boxes is None:
        print("  WARNING: no detector found >= 2 faces — using estimated positions")
        face_w = img_w // 5
        face_h = img_h // 3
        y_est  = int(img_h * 0.28)
        raw_boxes = [
            (img_w // 10,                y_est, img_w // 10 + face_w,              y_est + face_h),
            (img_w - img_w//10 - face_w, y_est, img_w - img_w//10,                y_est + face_h),
        ]

    # ── Extract tight head crops + record positions ───────────────────────────
    face_boxes = []
    paths      = []
    for i, (x1, y1, x2, y2) in enumerate(raw_boxes):
        hx1, hy1, hx2, hy2 = _face_to_head_crop(x1, y1, x2, y2, img_w, img_h)
        cropped = img_pil.crop((hx1, hy1, hx2, hy2))
        resized = cropped.resize((img_size, img_size), _lanczos)
        p = work_dir / f"speaker{i+1}.png"
        resized.save(p)
        paths.append(p)
        face_boxes.append([hx1, hy1, hx2, hy2])
        print(f"  speaker{i+1}.png: raw face ({x1},{y1},{x2},{y2}) "
              f"-> head crop ({hx1},{hy1},{hx2},{hy2}) [{hx2-hx1}×{hy2-hy1}px] "
              f"-> {img_size}px sq")

    (work_dir / "face_boxes.json").write_text(json.dumps(face_boxes, indent=2))
    return tuple(paths), face_boxes


def get_wav2lip_boxes(img_paths: list, work_dir: Path) -> list:
    """
    Get s3fd face boxes for each 512px head crop, with mtime-based caching.

    s3fd runs in a single subprocess for all speakers combined — one model load
    regardless of speaker count. On subsequent runs with the same images the cache
    is reused and s3fd is skipped entirely (~0s). Cache is invalidated automatically
    when the image file changes (new episode image).

    Returns list of [y1, y2, x1, x2] in 512px crop coordinates, one per speaker.
    """
    # Check which images have a valid cached box
    cached   = {}
    uncached = []
    for p in img_paths:
        cache_file = work_dir / f"{p.stem}_s3fd_box.json"
        if cache_file.exists():
            c = json.loads(cache_file.read_text())
            if c.get("mtime") == p.stat().st_mtime:
                cached[str(p)] = c["box"]
                print(f"  s3fd box (cached): {p.name} → {c['box']}")
                continue
        uncached.append(p)

    if uncached:
        print(f"  Running s3fd on {len(uncached)} image(s) — new/changed image, will cache...")
        img_list = repr([str(p) for p in uncached])
        script = (
            "import sys, json, numpy as np, cv2, torch\n"
            "sys.path.insert(0, '.')\n"
            "import face_detection\n"
            f"paths = {img_list}\n"
            "imgs = [cv2.imread(p) for p in paths]\n"
            "det = face_detection.FaceAlignment(\n"
            "    face_detection.LandmarksType._2D, flip_input=False, device='cuda')\n"
            "preds = det.get_detections_for_batch(np.array(imgs))\n"
            "del det; torch.cuda.empty_cache()\n"
            "boxes = []\n"
            "for r, img in zip(preds, imgs):\n"
            "    if r is None: sys.exit(1)\n"
            "    h, w = img.shape[:2]\n"
            "    boxes.append([max(0,int(r[1])), min(h,int(r[3])+15),\n"
            "                  max(0,int(r[0])), min(w,int(r[2]))])\n"
            "print(json.dumps(boxes))\n"
        )
        result = subprocess.run(
            [str(WAV2LIP_PYTHON), "-c", script],
            cwd=str(WAV2LIP_DIR), capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"s3fd detection failed:\n{result.stderr}")

        detected = json.loads(result.stdout.strip())
        for p, box in zip(uncached, detected):
            cache_file = work_dir / f"{p.stem}_s3fd_box.json"
            cache_file.write_text(json.dumps({"mtime": p.stat().st_mtime, "box": box}))
            cached[str(p)] = box
            print(f"  s3fd box (detected): {p.name} → {box}")

    return [cached[str(p)] for p in img_paths]


def animate(img_path: Path, audio_path: Path, output_path: Path, label: str,
            wav2lip_box: list):
    """
    Run Wav2Lip on a tight head crop using a pre-computed s3fd face box.

    --box skips Wav2Lip's internal s3fd load entirely. The box comes from
    get_wav2lip_boxes() which runs s3fd once per new image and caches the result.
    """
    print(f"Animating {label} (Wav2Lip)...")

    audio_16k = WORK_DIR / f"{img_path.stem}_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(audio_16k)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    y1, y2, x1, x2 = wav2lip_box
    cmd = [
        str(WAV2LIP_PYTHON),
        str(WAV2LIP_DIR / "inference.py"),
        "--checkpoint_path", str(WAV2LIP_DIR / "checkpoints/wav2lip_gan.pth"),
        "--face",    str(img_path),
        "--audio",   str(audio_16k),
        "--outfile", str(output_path),
        "--fps",     str(FPS),
        "--box",     str(y1), str(y2), str(x1), str(x2),
        "--nosmooth",
        "--wav2lip_batch_size", "256",
    ]
    subprocess.run(cmd, cwd=str(WAV2LIP_DIR), check=True)
    print(f"  {label}: {output_path.stat().st_size/1e6:.1f} MB")


def composite(orig_img: Path, vid1: Path, vid2: Path, audio: Path,
              output: Path, timeline: list, face_boxes: list):
    """
    Compose final video using orig_img as the full background.

    When speaker 0 talks: scale vid1 to face_boxes[0] size, overlay at that position.
    When speaker 1 talks: scale vid2 to face_boxes[1] size, overlay at that position.
    When silent: unmodified orig_img is shown — studio, BabyRage logo, everything intact.
    """
    print(f"Compositing (overlay mode, {len(timeline)} segments)...")

    (hx1_0, hy1_0, hx2_0, hy2_0) = face_boxes[0]
    (hx1_1, hy1_1, hx2_1, hy2_1) = face_boxes[1]
    ow0, oh0 = hx2_0 - hx1_0, hy2_0 - hy1_0
    ow1, oh1 = hx2_1 - hx1_1, hy2_1 - hy1_1

    def _enable_expr(speaker_idx):
        segs = [s for s in timeline if s["speaker"] == speaker_idx]
        if not segs:
            return "0"
        return "+".join(f"between(t,{s['start']:.4f},{s['end']:.4f})" for s in segs)

    enable0 = _enable_expr(0)
    enable1 = _enable_expr(1)

    # Feather: fade alpha from 0 at the edge to 1 over FEATHER pixels on all 4 sides.
    # Softens the hard boundary between the GAN-animated region and the static background.
    F = 20

    def _feather(w, h):
        return (
            f"format=rgba,"
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
            f"a='255*min(1,min(min(X/{F},({w}-1-X)/{F}),min(Y/{F},({h}-1-Y)/{F})))'"
        )

    filter_complex = (
        "[0:v]setpts=PTS-STARTPTS[bg];"
        f"[1:v]scale={ow0}:{oh0},setpts=PTS-STARTPTS,{_feather(ow0, oh0)}[face0];"
        f"[2:v]scale={ow1}:{oh1},setpts=PTS-STARTPTS,{_feather(ow1, oh1)}[face1];"
        f"[bg][face0]overlay={hx1_0}:{hy1_0}:enable='{enable0}'[v0];"
        f"[v0][face1]overlay={hx1_1}:{hy1_1}:enable='{enable1}'[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS), "-loop", "1", "-i", str(orig_img),  # 0: background
        "-i", str(vid1),   # 1: speaker 0 animated head
        "-i", str(vid2),   # 2: speaker 1 animated head
        "-i", str(audio),  # 3: audio
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output),
    ]
    subprocess.run(cmd, check=True)
    print(f"  Final video: {output} ({output.stat().st_size/1e6:.1f} MB)")


AUDIO_SRC   = _HERE.parent / "Input" / "Audio" / "Source"
IMAGE_SRC   = _HERE.parent / "Input" / "Image" / "Source"
OUTPUT_BASE = _HERE / "output"


def _find_first(folder: Path, exts: tuple) -> Path:
    for ext in exts:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file with extensions {exts} found in {folder}")


def main():
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",        default=None, help="Override audio path")
    parser.add_argument("--img",          default=None, help="Override image path")
    parser.add_argument("--output",       default=None, help="Override output path")
    parser.add_argument("--skip-diarize", action="store_true",
                        help="Reuse speaker WAVs + timeline from a previous run")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve() if args.audio else \
                 _find_first(AUDIO_SRC, (".wav", ".mp3", ".flac"))
    img_path   = Path(args.img).resolve()   if args.img   else \
                 _find_first(IMAGE_SRC, (".png", ".jpg", ".jpeg"))

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        dated_dir  = OUTPUT_BASE / datetime.now().strftime("%d.%m.%y")
        dated_dir.mkdir(parents=True, exist_ok=True)
        output_path = dated_dir / (audio_path.stem + ".mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Audio : {audio_path}")
    print(f"Image : {img_path}")
    print(f"Output: {output_path}")

    for p, name in [(audio_path, "audio"), (img_path, "image")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    if not (WAV2LIP_DIR / "checkpoints/wav2lip_gan.pth").exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found at {WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth\n"
            "Run setup_wav2lip.sh to download model weights."
        )

    if not args.skip_diarize:
        result       = diarize(audio_path, WORK_DIR)
        speaker_wavs = result["speaker_wavs"]
        timeline     = result["timeline"]
    else:
        speaker_wavs = [WORK_DIR / "speaker1.wav", WORK_DIR / "speaker2.wav"]
        for p in speaker_wavs:
            if not p.exists():
                raise FileNotFoundError(f"--skip-diarize: {p} not found. Run without flag first.")
        tl_path = WORK_DIR / "timeline.json"
        if not tl_path.exists():
            raise FileNotFoundError(f"--skip-diarize: {tl_path} not found. Run without flag first.")
        timeline = json.loads(tl_path.read_text())
        print(f"Skipping diarization — reusing WAVs + timeline ({len(timeline)} segments)")

    (img1, img2), face_boxes = extract_face_crops(img_path, WORK_DIR)
    wav2lip_boxes = get_wav2lip_boxes([img1, img2], WORK_DIR)

    vid1 = WORK_DIR / "speaker1_anim.mp4"
    vid2 = WORK_DIR / "speaker2_anim.mp4"

    # Animate sequentially (GPU-bound; parallel causes contention on a single GPU)
    animate(img1, speaker_wavs[0], vid1, "Speaker 1", wav2lip_boxes[0])
    animate(img2, speaker_wavs[1], vid2, "Speaker 2", wav2lip_boxes[1])

    composite(img_path, vid1, vid2, audio_path, output_path, timeline, face_boxes)
    print(f"\nDone! -> {output_path}")


if __name__ == "__main__":
    main()
