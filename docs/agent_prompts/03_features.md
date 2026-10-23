# Prompt 03 - Step 3: Feature extraction + caching (one model at a time)

Follow AGENTS.md. First produce the implementation plan and WAIT for my OK, then implement.
Requires Step 2 output in data/processed/.

## Goal
`python scripts/extract_features.py --all` reads every processed clip and writes cached encoder
outputs to `data/features/<clip_id>/`, loading ONE model at a time (load -> all clips -> free -> next),
skipping anything already cached, and printing time + peak memory per encoder.

## Read first
AGENTS.md, configs/default.yaml (`models`, `models.dims`), a processed clip's metadata.json,
src/preprocess/ (Step 2 code), placeholder files in src/encoders/.

## Downloads (ask first)
Before the first real run, list each model with its real download size (check the Hugging Face
repo file sizes; do not guess) and wait for my OK:
`openai/clip-vit-base-patch16`, `facebook/wav2vec2-base`, `FacebookAI/roberta-base`,
`openai-community/gpt2`. Each must be < 1 GB; if any is larger, say so.
Use the default Hugging Face cache locally (on Colab the notebook points HF_HOME to Drive).

## Config additions (configs/default.yaml, new section `features`)
```yaml
features:
  image_batch_size: 4        # colab: 32
  audio_chunk_s: 20          # wav2vec2 runs on <= 20 s windows to bound RAM
  text_max_tokens: 512
  detectgpt_max_tokens: 512
  detectgpt_min_tokens: 16   # shorter transcripts -> score NaN + flag "text_too_short"
  dtype: float32
```

## Implement

### src/utils/profiling.py
- Context manager `measure(name) -> Measurement` with `seconds` and `peak_rss_mb`: a background
  thread samples `psutil.Process().memory_info().rss` every 20 ms (works on Linux and Windows).
  Test it in tests/test_profiling.py (allocating a 200 MB numpy array raises the peak by >= 150 MB).

### src/encoders/image_clip.py
- `load_clip(name, device) -> (model, processor)`; `encode_images(images: list[np.ndarray], model,
  processor, device, batch_size) -> np.ndarray [N, dims.image]` (float32) under `torch.inference_mode()`.
- Use the CLIP *projected* image embedding (dims.image = 512). The return type of
  `get_image_features` changed across transformers versions: CHECK it in the installed
  transformers 5.18 and extract the tensor correctly. Assert the final shape.
- Per clip write: `face_clip.npy [16, 512]`, `mouth_clip.npy [16, 512]` (zero rows where no face),
  `face_mask.npy [16]` bool (from metadata faces.detected). Store raw, not L2-normalised.

### src/encoders/audio_w2v.py
- `load_w2v(name, device)`; `encode_audio(wave, sr, model, extractor, device, chunk_s) -> np.ndarray [T, dims.audio]`
  using `last_hidden_state`; chunk long audio into <= chunk_s windows and concatenate.
  wav2vec2 outputs about 50 frames per second, so a 10 s clip gives about 499 rows.
- `align_to_frames(feats [T, D], duration_s, n_frames=16) -> [16, D]`: frame i is the mean of audio
  frames inside segment `[i*dur/16, (i+1)*dur/16)`, the same segments Step 2 used for frame sampling.
- Per clip write: `audio_w2v.npy [T, 768]`, `audio_aligned.npy [16, 768]`. No audio -> write nothing,
  record `audio_present: false` in the manifest.

### src/encoders/text_encoder.py
- `load_text_encoder(name, device)`; `encode_text(text, model, tokenizer, device, max_tokens) -> [768]`
  = `last_hidden_state[:, 0]` (the `<s>`/CLS token). Empty transcript -> zeros + `text_present: false`.
- Write `text_roberta.npy [768]`.

### src/encoders/fast_detectgpt.py
- `sampling_discrepancy(logits_ref [T, V], logits_score [T, V], labels [T]) -> float` (Fast-DetectGPT,
  Bao et al., ICLR 2024, analytic version):
  ```
  lp      = log_softmax(logits_score)          # scoring model log-probs
  q       = softmax(logits_ref)                # reference/sampling model probs
  ll      = lp[t, labels[t]]                   # log-likelihood of the observed tokens
  mu      = sum_v q * lp                       # expected log-prob per position
  var     = sum_v q * lp**2 - mu**2
  score   = (sum_t ll - sum_t mu) / sqrt(sum_t var)
  ```
  Logits at position t predict token t+1: use `logits[:-1]` with `labels = ids[1:]`.
- `score_text(text, scoring, reference, tokenizer, device, max_tokens, min_tokens) -> dict(score, num_tokens, flag)`.
  If reference and scoring are the same model name, load it once and use the same logits for both.
- Higher score = more likely machine-written. Write `detectgpt.json {score, num_tokens, model, flag}`.

### scripts/extract_features.py
- Flags: common args + `--clip-id` (repeatable) or `--all`, `--encoders clip,w2v,roberta,detectgpt`
  (default all), `--overwrite`.
- Loop order: for each encoder: load model -> process every selected clip -> `del model; gc.collect()`
  (and `torch.cuda.empty_cache()` on cuda) -> next encoder. Never two big models in memory at once.
- Skip a clip/encoder when its output exists and has the expected shape (log "cached, skipped").
- Per clip `manifest.json`: for every output file: shape, dtype, model, seconds, created_at; plus
  `audio_present`, `text_present`, flags.
- Final table: encoder | clips done | skipped | total s | s per clip | peak RSS MB | output shape.

## Tests (no network, no big downloads)
Build tiny random models from config objects in the tests (e.g. CLIPConfig / Wav2Vec2Config /
RobertaConfig / GPT2Config with hidden size 32, 2 layers) instead of downloading.
- image: output shape `[N, projection_dim]` for N = 1 and 5 with batch_size 2 (batching gives the
  same result as one batch, atol 1e-5).
- audio: chunked vs unchunked on 5 s of noise -> same T (+-1) and close values; `align_to_frames`
  shape [16, D] and correct means on a hand-made array.
- `sampling_discrepancy`: matches a plain-numpy re-implementation on random logits (atol 1e-5);
  when every label is the argmax token the score is clearly positive; random labels -> lower score.
- extract script on a fake processed clip with tiny models injected: writes files with the expected
  shapes; second run skips; `--overwrite` recomputes.
- Tests that need real tokenizers or real models: `@pytest.mark.network`.

## Demo commands to run and show
```
<venv python> -m pytest -rs
<venv python> scripts/extract_features.py --all --device cpu
<venv python> scripts/extract_features.py --all          (second run: everything skipped, fast)
```
Show: the final table, one clip's manifest.json, a python one-liner printing every .npy shape,
the detectgpt score, and (sanity check) the mean cosine similarity between consecutive face embeddings.

## Acceptance criteria
- [ ] Shapes: face_clip (16, 512), mouth_clip (16, 512), face_mask (16,), audio_w2v (about 50 x seconds, 768),
      audio_aligned (16, 768), text_roberta (768,), detectgpt score finite (or NaN + flag for short text).
- [ ] Exactly one large model in memory at any time (show peak RSS per encoder; each < 3 GB on this laptop).
- [ ] Second run skips everything and finishes in seconds.
- [ ] Real timings per encoder on this laptop reported (seconds per clip).
- [ ] All tests pass; network tests skipped or passing, explained.

STOP after the RESULTS REPORT. Proposed commit:
`"Step 3: feature extraction + caching (CLIP, wav2vec2, RoBERTa, Fast-DetectGPT)"`
