"""
DeepFusionGuard - Step 1: preprocessing pipeline (Oct 2026 milestone)

For every video clip:
  FFmpeg A/V split -> 16 frames -> MTCNN face + mouth crop (224x224)
  -> 16 kHz mono audio -> (optional) Whisper transcript

Output per clip <id>:
  <out>/<id>.npz   face (16,224,224,3) uint8, mouth (16,224,224,3) uint8
  <out>/<id>.wav   16 kHz mono audio
  <out>/<id>.txt   transcript (only with --transcribe)
and one row per clip in <out>/manifest.csv

Usage:
  python preprocess.py --input_dir data/FakeAVCeleb --output_dir cache/fakeavceleb \
      --label_from_path --limit 20
"""
import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

N_FRAMES = 16
SIZE = 224
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def extract_audio(video: Path, wav_out: Path) -> bool:
    """FFmpeg: demux audio to 16 kHz mono wav. Returns False if no audio stream."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", "16000", str(wav_out)]
    res = subprocess.run(cmd)
    return res.returncode == 0 and wav_out.exists() and wav_out.stat().st_size > 0


def sample_frames(video: Path, n: int = N_FRAMES):
    """Uniformly sample n RGB frames from the video."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def crop_square(img, cx, cy, half):
    """Crop a square around (cx, cy) with border clamping, resize to SIZE."""
    h, w = img.shape[:2]
    x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x2, y2 = int(min(cx + half, w)), int(min(cy + half, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (SIZE, SIZE))


def face_and_mouth(mtcnn, frame, prev):
    """Return (face_crop, mouth_crop, state). Falls back to prev detection, then centre crop."""
    boxes, probs, pts = mtcnn.detect(frame, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        b, p = boxes[0], pts[0]                     # most confident face
        prev = (b, p)
    if prev is None:                                # nothing ever detected
        h, w = frame.shape[:2]
        half = min(h, w) // 2
        c = crop_square(frame, w // 2, h // 2, half)
        return c, c, None
    b, p = prev
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    face = crop_square(frame, cx, cy, max(b[2] - b[0], b[3] - b[1]) / 2 * 1.2)
    mx, my = p[3:5, 0].mean(), p[3:5, 1].mean()    # mid-point of mouth corners
    mouth = crop_square(frame, mx, my, (b[2] - b[0]) * 0.35)
    return face, mouth, prev


def transcribe(asr, wav: Path) -> str:
    return asr(str(wav), return_timestamps=False)["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--transcribe", action="store_true", help="run Whisper-small ASR")
    ap.add_argument("--label_from_path", action="store_true",
                    help="label=fake if 'fake' appears in the path, else real (check per dataset!)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N clips (testing)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device)

    asr = None
    if args.transcribe:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                       device=0 if device == "cuda" else -1)

    videos = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if args.limit:
        videos = videos[: args.limit]

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "src_path", "label", "has_audio", "n_frames", "face_found"])
        for v in tqdm(videos, desc="clips"):
            cid = v.stem
            frames = sample_frames(v)
            if len(frames) < N_FRAMES // 2:
                continue                              # unreadable / too short
            while len(frames) < N_FRAMES:             # pad by repeating last frame
                frames.append(frames[-1])

            faces, mouths, prev, found = [], [], None, 0
            for fr in frames:
                face, mouth, prev = face_and_mouth(mtcnn, fr, prev)
                found += int(prev is not None)
                faces.append(face)
                mouths.append(mouth)
            if any(x is None for x in faces + mouths):
                continue

            np.savez_compressed(out / f"{cid}.npz",
                                face=np.stack(faces), mouth=np.stack(mouths))

            wav = out / f"{cid}.wav"
            has_audio = extract_audio(v, wav)
            if asr and has_audio:
                (out / f"{cid}.txt").write_text(transcribe(asr, wav))

            label = int("fake" in str(v).lower()) if args.label_from_path else -1
            w.writerow([cid, str(v), label, int(has_audio), len(frames), found])


if __name__ == "__main__":
    main()
