# Prompt 02 - Step 2: Preprocessing pipeline

Follow AGENTS.md. First produce the implementation plan and WAIT for my OK, then implement.

## Goal
`python scripts/preprocess.py --input data/samples/x.mp4` turns one video into everything later
steps need, saved under `data/processed/<clip_id>/`: 16 kHz mono audio, 16 evenly spaced frames,
224x224 face + mouth crops, a transcript, a metadata.json, and a preview image a human can check.

## Read first
AGENTS.md, configs/default.yaml (section `preprocess`), src/utils/config.py, src/utils/cli.py,
scripts/show_config.py (script pattern), the placeholder files in src/preprocess/.

## Inputs
Sample videos in `data/samples/` (not committed). If it is empty, STOP and ask me for clips.
Recommended: 2-3 phone recordings, 5-15 s, one teammate speaking facing the camera, plus
optionally one clip with no face (e.g. filming a wall) to test the no-face path.

## Implement (keep these exact names; add more helpers if useful)

### src/preprocess/video_split.py
- `probe_video(path: Path) -> dict` via `ffprobe -v error -print_format json -show_streams -show_format`:
  returns `duration_s, fps, width, height, num_frames (may be None), has_audio, video_codec, audio_codec`.
  Raise `FileNotFoundError` for a missing file and `RuntimeError("no video stream in <file>")`.
- `extract_audio(video: Path, out_wav: Path, sample_rate: int) -> Path | None`:
  `ffmpeg -y -v error -i <video> -vn -ac 1 -ar <sr> -c:a pcm_s16le <out_wav>`.
  Return None (and log a warning "audio stream missing in <file>") when the video has no audio.
  Use `subprocess.run(..., check=True, capture_output=True, text=True)`; on failure raise
  RuntimeError including ffmpeg's stderr.

### src/preprocess/frames.py
- `sample_frame_indices(total_frames: int, n: int) -> list[int]`: segment centers,
  `idx_i = floor((i + 0.5) * total_frames / n)`, strictly increasing, all in `[0, total_frames-1]`.
  If `total_frames < n`, repeat indices (still length n) and log a warning.
- `read_frames(video: Path, indices: list[int]) -> list[np.ndarray]` (RGB, uint8, HxWx3) with
  cv2.VideoCapture. Container frame counts can be wrong: if seeking fails or CAP_PROP_FRAME_COUNT
  is 0, fall back to decoding sequentially. Never return fewer than len(indices) frames: if a frame
  cannot be read, raise RuntimeError("could not read frame 12 of <file>").
- Return the timestamps too: `timestamps_s = idx / fps`.

### src/preprocess/face_crop.py
- `ensure_yunet_model(cfg) -> Path`: model file `preprocess.yunet_model` in `get_path(cfg, "model_cache")`.
  If missing, download from `preprocess.yunet_url`. Before downloading, check its size; it should be small
  (expected well under 1 MB). If it is larger than 5 MB, stop and ask. After downloading, verify that
  `cv2.FaceDetectorYN.create` loads it.
- `class FaceDetector`: wraps `cv2.FaceDetectorYN`; `detect(frame_rgb) -> Detection | None` where
  `Detection` is a dataclass `bbox (x, y, w, h), landmarks (5x2 float array), score`.
  YuNet returns rows of 15 values: bbox(4), right eye(2), left eye(2), nose tip(2), right mouth
  corner(2), left mouth corner(2), score(1); VERIFY this order by printing one row from a real
  sample and checking the mouth points visually in the preview. Call `setInputSize` per frame size.
  Downscale frames whose longer side > 640 px for detection, then map coordinates back.
  If several faces: pick the largest box (tie -> highest score). Below `face_score_threshold` -> None.
- `crop_face(frame, det, margin, size) -> np.ndarray`: square box around bbox enlarged by
  `face_margin`, clamped to the image (pad with black if it goes outside), resized to `face_size`.
- `crop_mouth(frame, det, box_scale, size) -> np.ndarray`: square box centred on the midpoint of the
  two mouth corners, side = `mouth_box_scale` x corner distance, resized to `mouth_size`.
- No face in a frame: log `"no face found in frame 7 of <clip_id>"`, record it, continue.

### src/preprocess/audio.py
- `load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]`: float32 mono in [-1, 1]
  via soundfile; resample with librosa only if the rate differs.
- `audio_stats(wave, sr) -> dict`: duration_s, rms, peak, is_silent (rms < 1e-4).

### src/preprocess/transcript.py
- `transcribe(wav: Path, cfg, device: str) -> dict` with faster-whisper: model `preprocess.transcript.model`,
  compute_type from config (int8 on CPU; if device is cuda use the colab value). Return
  `text, language, language_probability, segments [{start, end, text}], model, duration_s`.
- Load the model inside the function and free it afterwards (`del model; gc.collect()`).
- Silent or missing audio -> return empty text with `"skipped_reason"`, do not crash.
- First run downloads the Whisper model (base is about 150 MB; check and report the real size).

### scripts/preprocess.py
- Flags: common args (add_common_args) + `--input` (file or folder of videos), `--clip-id` (single
  file only; default = sanitized file stem: lowercase, `[a-z0-9_-]` only), `--skip-transcript`,
  `--overwrite`.
- If `processed/<clip_id>/metadata.json` exists and no `--overwrite`: print "cached, skipped" and continue.
- Output layout:
  ```
  data/processed/<clip_id>/
    audio.wav                  16 kHz mono PCM (absent if the video has no audio)
    frames/frame_00.jpg ... frame_15.jpg
    faces/face_00.png ...      only for frames with a face, 224x224
    mouths/mouth_00.png ...    only for frames with a face, 224x224
    transcript.json
    metadata.json
    preview.png                4 rows: frames / faces / mouths (grey tile if no face) / waveform
  ```
- metadata.json (exact keys): `clip_id, source_path, source_sha256, created_at, profile, config_hash,
  video {duration_s, fps, width, height, num_frames, codec}, frames {count, indices, timestamps_s},
  faces {detected: [bool x16], scores: [float|null x16], num_face_frames, detector},
  audio {has_audio, sample_rate, duration_s, rms, is_silent}, transcript {model, language, num_chars},
  flags [list of strings], timings_s {probe, audio, frames, faces, transcript, total},
  versions {python, opencv, ffmpeg, faster_whisper}`.
- Flags to set: `"no_audio"`, `"silent_audio"`, `"few_faces"` (num_face_frames < min_face_frames),
  `"empty_transcript"`.
- End of run: a summary table, one line per clip: clip_id, duration, faces x/16, has_audio,
  transcript chars, flags, total seconds. Exit code 0 if all clips processed, 1 if any failed
  (failures are logged with the reason and do not stop the other clips).

## Tests (tests/test_preprocess.py)
Make synthetic videos in a pytest tmp folder with ffmpeg (no network, no real faces):
`ffmpeg -f lavfi -i testsrc=size=320x240:rate=25 -f lavfi -i sine=frequency=440:sample_rate=44100 -t 3 -c:v mpeg4 -c:a aac -shortest clip.mp4`
and a second one without the sine input (no audio). Use `-c:v mpeg4` (built into every ffmpeg).
- `sample_frame_indices`: length 16, increasing, in range, for totals 16, 75, 300, 10 (repeat case).
- `probe_video` values for the synthetic clip (fps 25, 320x240, duration ~3 s, has_audio True/False).
- `extract_audio`: output is mono, 16000 Hz, duration within 0.1 s of 3 s; None for the no-audio clip.
- `read_frames`: 16 RGB uint8 arrays of shape (240, 320, 3).
- `crop_face` / `crop_mouth` on a synthetic Detection: output exactly (224, 224, 3), also when the
  box sticks out of the image.
- End-to-end `scripts/preprocess.py` on the synthetic clip with `--skip-transcript`: metadata.json has
  all keys, faces.detected all False, flags contain "few_faces"; second run prints "cached, skipped".
- One test on a real sample (skip if data/samples/ is empty): at least 8/16 faces detected.
- Transcription test marked `@pytest.mark.network` (needs the Whisper download).
Add the new files to tests/test_structure.py.

## Demo commands to run and show
```
<venv python> -m pytest -rs
<venv python> scripts/preprocess.py --input data/samples/ --device cpu
<venv python> scripts/preprocess.py --input data/samples/        (second run -> "cached, skipped")
```
Then show: the folder tree of one processed clip, its full metadata.json, its transcript text,
and open its preview.png so I can see it (describe what it shows: are the mouth crops centred on the mouth?).

## Acceptance criteria
- [ ] Every sample clip -> 16 frames, 16 timestamps increasing, audio.wav 16000 Hz mono.
- [ ] Speaking-face clips: >= 14/16 faces detected; mouth crops visibly centred on the lips in preview.png.
- [ ] No-face clip (if provided): processed without crashing, flag `few_faces`, log lines name the frames.
- [ ] Transcript text matches what was said (state language detected + first 100 characters).
- [ ] Second run skips cached clips; `--overwrite` recomputes.
- [ ] Timing per stage reported for a 10 s clip on this laptop (real numbers from metadata).
- [ ] All tests pass; new skips explained.

STOP after the RESULTS REPORT. Proposed commit:
`"Step 2: preprocessing pipeline (ffmpeg split, frames, YuNet face/mouth crops, transcript)"`
