# Prompt 04 - Step 4: Core model modules (pure PyTorch, tested with random tensors)

Follow AGENTS.md. First produce the implementation plan and WAIT for my OK, then implement.
No real data or downloads in this step: everything is tested with random tensors on CPU.

## Read first
AGENTS.md, configs/default.yaml (`fusion`, `models.dims`), placeholder files in src/fusion/,
src/consistency/, src/encoders/video_temporal.py.

## Notation and inputs (batch-first everywhere)
B = batch, T = 16 frames, d = fusion.d_model = 256, modalities M = [video, audio, text] (fusion.modalities).
Model input is a dict (what the Step 5 dataset will return):
```
face        [B, T, dims.image]   float   CLIP face embeddings
mouth       [B, T, dims.image]   float   CLIP mouth embeddings
frame_mask  [B, T]               bool    True = face present in that frame
audio       [B, T, dims.audio]   float   wav2vec2 features aligned to the 16 frames
text        [B, dims.text]       float   RoBERTa CLS embedding
detectgpt   [B]                  float   Fast-DetectGPT score (NaN allowed -> treated as missing)
present     [B, 3]               bool    modality available: [video, audio, text]
```

## Implement

### src/fusion/projection.py (new file)
`ModalityProjector(in_dim, d_model, dropout)`: Linear -> LayerNorm -> GELU -> Dropout. Works on
`[B, D]` and `[B, T, D]`.

### src/encoders/video_temporal.py
`TemporalTransformer(d_model, num_layers=fusion.temporal_layers, num_heads, max_len=16, dropout)`:
learned positional embedding + `nn.TransformerEncoder` (batch_first, norm_first=True,
dim_feedforward = 4*d). Uses `src_key_padding_mask = ~frame_mask`. If a sample has no valid frame,
do not pass an all-masked row to attention (NaN risk): unmask it and zero the output instead.
Output `[B, T, d]`.

### src/fusion/cross_attention.py
`CrossModalBlock(d_model, num_heads, dropout)`:
- `forward(query [B, Tq, d], query_mask [B, Tq], context [B, Tk, d], context_mask [B, Tk]) -> pooled [B, d]`
- `attn = MultiheadAttention(Q=query, K=V=context, key_padding_mask=~context_mask)`, i.e.
  `CrossAttn = softmax(Q K^T / sqrt(d_k)) V` with d_k = d / num_heads.
- `x = LayerNorm(query + Dropout(attn))`; `x = LayerNorm(x + Dropout(FFN(x)))` (FFN d -> 4d -> d, GELU).
- Samples whose context is fully masked (no other modality present): skip attention for them
  (use `x = LayerNorm(query)`); there must be NO NaN anywhere.
- Masked mean pooling over valid query positions -> `[B, d]`.
Context for modality m = concatenation of the tokens of all OTHER modalities (masks concatenated
the same way; a missing modality contributes fully-masked tokens).

### src/consistency/sync.py
`sync_score(mouth [B, T, d], audio [B, T, d], frame_mask [B, T]) -> [B]`:
`s_sync = 1 - mean_t cos(mouth_t, audio_t)` over frames where `frame_mask` is True; range [0, 2].
Inputs are the PROJECTED tokens (common d), so the module is trainable end-to-end.
Return 0 for samples missing video or audio (the caller passes a mask).

### src/consistency/emotion.py
`emotion_mismatch(...) -> [B]` returns zeros. Docstring + `# TODO(Phase 2): s_emo = KL(p_face || p_voice)`.

### src/consistency/authorship.py
`authorship_score(detectgpt [B], center, scale) -> [B]` = `sigmoid((score - center) / scale)`,
NaN -> 0. Add to config: `consistency: {detectgpt_center: 0.0, detectgpt_scale: 1.0}` with a comment
"to be calibrated on HC3 in a later step".

### src/consistency/__init__.py
`consistency_vector(s_sync, s_emo, s_auth) -> c [B, 3]`.

### src/fusion/gated_fusion.py
`GatedFusion(d_model, num_modalities=3, c_dim=3, modality_dropout=fusion.modality_dropout)`:
```
g_m   = sigmoid(w_g^T [h_m ; c] + b_g)      one shared Linear(d + 3, 1) for all modalities
g_m   = 0 if modality m is missing (present mask)
train mode: modality dropout - each PRESENT modality is dropped with p = 0.3 per sample;
            if that would drop all present modalities of a sample, keep one present modality
            (chosen at random); dropped -> g_m = 0
alpha_m = g_m / sum_j g_j      (eps 1e-8; at least one g > 0 is guaranteed)
z       = sum_m alpha_m * h_m  -> [B, d]
```
Return `z, alpha [B, M], gates [B, M]`. Eval mode: no modality dropout (deterministic).

### src/fusion/classifier.py
`Classifier(d_model=256, hidden=fusion.classifier_hidden=128, dropout=fusion.classifier_dropout)`:
Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 1) -> logits `[B]` (sigmoid applied only at inference/metrics).

### src/fusion/model.py (new file)
`DeepFusionGuard(cfg)` wires everything:
1. Project: face, mouth -> d (separate projectors), audio -> d, text -> d as one token `[B, 1, d]`.
2. Video tokens = TemporalTransformer(projected face, frame_mask).
3. For each modality m: `h_m = CrossModalBlock_m(tokens_m, mask_m, other tokens, other masks)`
   (one block per modality).
4. `c = [sync_score(proj mouth, proj audio), emotion_mismatch, authorship_score(detectgpt)]`,
   with s_sync = 0 where video or audio is missing.
5. `z, alpha, gates = GatedFusion(h, c, present)`; `logits = Classifier(z)`.
Return a dict: `logits [B], prob [B], alpha [B, 3], gates [B, 3], c [B, 3]`, plus
`contribution_pct = alpha * 100` (what the dashboard shows).
Add a `count_parameters(model) -> dict` helper and print it in the demo.

## Tests (tests/test_fusion.py, tests/test_consistency.py) - all with random tensors, seeded
- Shapes for B = 1 and B = 4 (logits [B], alpha [B, 3], c [B, 3]).
- alpha sums to 1 (atol 1e-5) for every sample, in train and eval mode.
- Missing modality => its alpha is exactly 0 (each of the 3 modalities, and pairs).
- Only one modality present => its alpha == 1, outputs finite (no NaN), for each modality.
- Modality dropout: in train mode over 200 forward passes, no sample ever has all alphas 0, and the
  observed drop rate of present modalities is about 0.3 (between 0.2 and 0.4); eval mode -> two calls
  give identical outputs.
- Frames with frame_mask False do not change the output (change their values -> same result in eval).
- sync_score: identical mouth/audio -> 0; mouth = -audio -> 2; masked frames ignored.
- authorship_score: NaN -> 0; monotonic in the score.
- Backward pass with all modalities present: every trainable parameter gets a finite non-zero gradient
  (list any that do not and explain).
- Speed sanity: one eval forward with B = 4 on CPU in < 1 s (report the real time).
Add new files to tests/test_structure.py.

## Demo
`scripts/model_demo.py` (common args): builds the model from config, runs one batch of random
inputs with modalities missing in different rows, and prints a table per sample:
present [v, a, t] | contribution % video / audio / text | c = [s_sync, s_emo, s_auth] | prob.
Also prints the parameter count per component and in total.

## Acceptance criteria
- [ ] All tests above pass (show the pytest summary).
- [ ] Demo table shows 0.0 % contribution for every missing modality and rows summing to 100 %.
- [ ] No NaN in any output, including rows with a single modality.
- [ ] Parameter count reported (expect a few million; report the exact number).
- [ ] Formulas in code match this prompt (quote the lines computing g_m, alpha_m, z, s_sync).

STOP after the RESULTS REPORT. Proposed commit:
`"Step 4: core model modules (projection, temporal transformer, cross-attention, consistency, gated fusion, classifier)"`
