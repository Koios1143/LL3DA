# Extra benchmarks for LL3DA

The released generalist checkpoint
[`CH3COOK/LL3DA-weight-release/ll3da-opt-1.3b.pth`](https://huggingface.co/CH3COOK/LL3DA-weight-release)
(OPT-1.3B, frozen Vote2Cap-DETR detector) evaluated on **Scan2Cap**, **ScanQA**, **SQA3D**,
**Real-3DQA**, **Beacon3D QA (ScanNet, 3RScan, MultiScan)** and **XR-QA / XR-SceneCaption**
(zero-shot and finetuned) with the repo's own model,
prompts, generation (beam 4) and scorers (XR: LSceneLLM's protocol, see its section).

Not evaluated: **ScanRefer, Multi3DRefer and Beacon3D grounding**. The released generalist only
generates text: it was trained with `--max_prompt 1`, i.e. only the first prompt of each task,
none of which asks for locations, and the box-outputting `tuning.ovdet.sh` model is not
released. Grounding would need a protocol LL3DA was never trained or evaluated with.

## Run

```bash
cp env.sh.example env.sh            # fill in paths (+ OPENAI_API_KEY for Beacon3D QA)
bash extra_eval/build_env.sh        # uv venv: torch 2.7.1+cu128, transformers 4.40.2, pointnet2 + cython ops
bash extra_eval/download.sh         # checkpoint + preprocessed ScanNet (scannet_data.zip), OPT-1.3B, BERT embedding
PREFIX_CACHE=1 bash extra_eval/run_infer.sh   # GPU; one process per visible GPU; all benchmarks
                                              # (drop PREFIX_CACHE=1 for the repo's own generation code)
bash extra_eval/score.sh            # CPU; Java for METEOR
```

The README's torch 1.13.1+cu116 has no Blackwell (sm_120) kernels, hence the newer stack; the
model code is the repo's, unchanged. `extra_eval/infer.py` replaces `main.py --test_only` so
that all benchmarks run in one pass and only the val sets are built (`main.py` also loads
every training set).

**Inputs.** As the repo's val datasets build them: the scene's `_aligned_vert.npy` (xyz, rgb,
normal + height) subsampled to 40k points, the first prompt of `TASK_PROPMT['qa']` /
`['densecap']`, instructions padded to 256 tokens (`scripts/opt-1.3b/eval.*.sh`). One change:
the 40k-point subsample is seeded by the scene name, so every question of a scene sees the same
points and runs are reproducible (upstream draws a new subsample per item).

## Results

### Paper benchmarks (ScanNet val)

The paper's Table 5, last row ("w/o fine-tuning" = the released generalist), scored by the
repo's own `utils/capeval` (no PTB tokenizer) as `eval_utils/evaluate_{densecap,qa}.py` do.

| Benchmark | Metric | Ours | Paper |
|---|---|---:|---:|
| Scan2Cap (ScanRefer val, 2068 objects) | CIDEr@0.5 | 63.19 | 62.98 |
| | BLEU-4@0.5 | 35.19 | 35.97 |
| | METEOR@0.5 | 25.69 | 25.66 |
| | ROUGE-L@0.5 | 54.02 | 54.65 |
| ScanQA val (4675) | CIDEr | 75.10 | 75.67 |
| | BLEU-4 | 13.25 | 13.33 |
| | METEOR | 15.24 | 15.37 |
| | ROUGE-L | 36.56 | 37.02 |

Scan2Cap at IoU 0.25: CIDEr 72.57, BLEU-4 40.02, METEOR 27.53, ROUGE-L 58.73 (1721 / 2068
objects have a proposal above 0.25, 1503 above 0.5). ScanQA EM / EM-R: 15.87 / 34.40.
[`results/scan2cap/`](results/scan2cap), [`results/scanqa/`](results/scanqa)

**Scan2Cap protocol** (the repo's `evaluate_densecap`): the generalist captions every
Vote2Cap-DETR proposal kept by objectness + NMS; each ScanRefer val object takes the caption of
its highest-IoU proposal above the threshold (`sos eos` if none) and is scored against all its
ScanRefer descriptions. This is LL3DA's own detector, not the Mask3D proposals of the other
baselines' Scan2Cap.

### SQA3D test (3519)

LL3DA has no SQA3D training or prompt; questions use its ScanQA prompt with
`<situation> <question>` inside the quotes. EM / EM-R by Real-3DQA's official `answer_match`
(exact / either-substring match after `clean_answer`, as for the other baselines).
[`results/sqa3d_test/`](results/sqa3d_test)

| EM | EM-R | What | Is | How | Can | Which | Others |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 29.89 | 33.96 | 14.91 | 42.18 | 23.87 | 52.96 | 36.47 | 33.22 |

(per-type columns: EM)

### Real-3DQA

HF `Oliver-Ma/Real-3DQA`, the same SQA3D-style prompt, scored by the official
`evaluate_debiased_exact_match.py` and `evaluate_rotation_robustness.py` (vendored in
`real3dqa/`, unmodified). The rotation splits change only the situation text.
[`results/real3dqa/`](results/real3dqa)

| Split | EM | EM-R |
|---|---:|---:|
| debiased_test (1485) | 15.76 | 19.73 |
| rotation 0 / 90 / 180 / 270 (750 each) | 7.33 / 6.13 / 5.33 / 7.07 | 12.00 / 9.73 / 10.80 / 11.60 |

| Rotation consistency | One | Two | Three | Four | VRS |
|---|---:|---:|---:|---:|---:|
| | 36.93 | 6.40 | 0.67 | 0.13 | 11.03 |

### Beacon3D QA

Official `evaluate_qa.py` (vendored in `beacon3d/Beacon3D`, unmodified), GPT judge
**`gpt-4o-mini-2024-07-18`**, ScanQA prompt. ScanNet scenes come from LL3DA's own
`scannet_data`; 3RScan / MultiScan are built the same way from the raw meshes by
`prepare_scans.py` (mesh vertices + face normals, z up, walls rotated onto x / y as ScanNet's
`axis_align_matrix` does, floor at 0, 50k points; 3RScan colours sampled from the textured mesh,
since the labelled ply's colours are instance colours). The detector is trained on ScanNet only.

| Domain (questions) | GPT-score (case) | object | class | app. | geo. | spatial | exist. | func. | EM | EM-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ScanNet (1155) | 44.91 | 6.75 | 22.30 | 40.47 | 55.70 | 47.21 | 51.12 | 66.67 | 41.30 | 43.38 |
| 3RScan (816) | 41.30 | 0.37 | 36.41 | 36.48 | 46.14 | 40.72 | 45.50 | – | 38.85 | 41.42 |
| MultiScan (477) | 25.52 | 1.26 | 16.94 | 22.81 | 32.29 | 25.74 | 31.57 | – | 22.01 | 24.95 |
| pooled (2448) | 39.93 | 3.55 | 23.74 | 35.69 | 50.17 | 40.63 | 45.03 | 66.67 | 36.72 | 39.13 |

[`results/beacon3d_*/`](results) (`beacon3d_overall/` also has the mean of the domains). No
grounding run, so the grounding-QA chain analysis is not reported.

### XR-QA / XR-SceneCaption (LSceneLLM, val)

[LSceneLLM](https://arxiv.org/abs/2412.01292)'s large-scene benchmarks: an XR scene is the
union of several SceneVerse HM3D regions (3 for QA, 5 for captions). LL3DA is in LSceneLLM's
Table 1, both zero-shot and finetuned. Scores are CIDEr / ROUGE-L / METEOR from LSceneLLM's
capeval (the same code as this repo's `utils/capeval`), keyed by its episode ids (2,981 QA keys,
283 captions), raw text as LSceneLLM validates.

```bash
SPLITS="train val" bash extra_eval/xr/prepare_xr.sh   # per-region points + normals -> $XR_CACHE
bash extra_eval/xr/run_xr.sh                          # zero-shot, both placements
bash extra_eval/xr/run_xr_finetune.sh                 # finetune, one placement per visible GPU
```

**Inputs.** LL3DA needs normals (`use_normal`); SceneVerse regions are points only, so
`prepare_xr.py` estimates them (PCA over 16 neighbours, oriented to the region's centre; the
same normals as Open3D's). Two placements of the regions: **rc** as LSceneLLM's loader
(region + `room_center`) and **glb** the real house layout (offsets registered against the
HM3D v0.2 meshes). 40k points per scene, seeded per scene.

**Protocol (`--xr_protocol`).** The paper "re-implements LL3DA keeping all other settings the
same as ours", i.e. inside LSceneLLM's framework. Only that reproduces its zero-shot row
(`lscenellm`, the default): the scene is centred and scaled into the unit sphere (`pc_norm`),
and the question goes into LSceneLLM's `hd_scene_qa` prompt as written. LL3DA's own setup
(`ll3da`: metric coordinates, its ScanQA prompt, lower-cased question) scores QA ~10 CIDEr
higher. Zero-shot QA, rc: `ll3da` 34.40, unit coords + ScanQA prompt 33.50, metric +
`hd_scene_qa` 27.00, `lscenellm` 24.52 — the prompt accounts for most of the gap.
Captions use LL3DA's chat decoding (`predict_chat`: top-k 50 / top-p 0.95 sampling, seeded
per scene); with beam search (`CAP_DECODE=beam`) they are ~130–180 words and lower in CIDEr.

| Setting | XR-QA | XR-SceneCaption |
| --- | --- | --- |
| zero-shot, rc | 24.52 / 11.87 / 5.76 | 0.49 / 10.97 / 5.55 |
| zero-shot, glb | 24.81 / 11.91 / 5.65 | 0.31 / 10.99 / 5.94 |
| zero-shot, paper | 24.78 / 12.66 / 5.31 | 0.12 / 8.71 / 5.14 |
| finetuned, rc, epoch 3 | **114.38 / 36.34 / 18.58** | 1.52 / 17.12 / 12.11 |
| finetuned, glb, epoch 3 | 114.85 / 36.75 / 18.74 | 1.42 / 17.22 / 12.04 |
| finetuned, paper | 112.80 / 36.94 / 18.68 | 3.22 / 20.95 / 13.49 |

- **Finetuning** (`finetune_xr.py`) follows the repo's generalist training: detector and
  OPT-1.3B frozen, the rest trained (111.0 M) with `CaptionNet`'s loss, AdamW (wd 0.1),
  per-iteration cosine to 1e-6, no warm-up, gradient clipping 0.1, batch 32. Data: XR data
  only (the paper adds LL3DA's own data): XR-QA train + XR-SceneCaption train, one item per
  answer (19,440), in the `lscenellm` protocol, a fresh 40k-point sample per item; 3 epochs
  (~20 min each on one GPU, including evaluation). lr 1e-4 (the repo's); LSceneLLM's own 1e-5
  reaches only 102.91 / 33.84 / 17.66 on QA after 3 epochs (`finetune_rc_lr1e-5/`).
- **QA matches the paper; captions stay below it** (best: rc epoch 2, 1.94 / 17.75 / 12.85).
  The missing LL3DA training data is a likely cause; not tested.
- Placement barely matters (≤ 0.5 CIDEr on QA after finetuning). Scenes with > 100
  SceneVerse objects score like the rest.

Predictions and scores (all, ≤ 100 / > 100 objects; `raw` and `ll3da`, which lower-cases the
caption references as `evaluate_scene_description.py` does):
[`results/xr/`](results/xr) (`zeroshot/`, `zeroshot_ll3da_protocol/`,
`finetune_{rc,glb}/epoch<k>/`, `finetune_rc_lr1e-5/` scores only; `*_beam` = beam-search
captions), summary in [`results/xr/summary.json`](results/xr/summary.json).

## Faster evaluation: `PREFIX_CACHE=1`

The repo's generation (`models/ll3da/generation_utils.py`) runs a beam search per question
(`predict_answer` loops over the batch) and re-runs OPT-1.3B over the whole sequence at every
step; dense captioning also runs the Q-Former once per proposal. LL3DA's Q-Former reads the
instruction, so its 32 scene tokens differ per question and there is no shared LLM prefix
across questions; what is shared per scene is the detector. `infer.py --fast`:

* groups questions by scene and runs the detector once per scene;
* runs the Q-Former batched over the scene's questions (dense captioning: its proposals);
* decodes 32 sequences at a time with `beam_search_kv`, the repo's `beam_search_decode` step for
  step (same scores, beam bookkeeping, heap of finished beams, forced end) but feeding only the
  newest token per beam against a KV cache reordered with the surviving beams; questions of
  different lengths are left-padded with an attention mask (OPT's positions follow the mask).

**Exactness.** With the LLM in float32, the fast path reproduces the repo's generation exactly:
219 / 219 ScanQA answers (one or 32 sequences per beam search) and 69 / 69 captions identical
(`check_fast_fp32.py`). In the released fp16, rounding differs between incremental and full
recomputation and flips beam near-ties: identical answers on 93–99% of questions per task and
74% of captions (`compare_preds.py`; the same rates with one sequence per beam search, so
batching/padding is not the cause). Scores move by at most a few tenths
([`results/prefix_cache/agreement.json`](results/prefix_cache/agreement.json)):

| | default | `PREFIX_CACHE=1` |
|---|---:|---:|
| Scan2Cap CIDEr@0.5 / B-4@0.5 | 63.19 / 35.19 | 63.05 / 35.44 |
| ScanQA CIDEr / B-4 / EM | 75.10 / 13.25 / 15.87 | 75.36 / 12.96 / 15.87 |
| SQA3D test EM / EM-R | 29.89 / 33.96 | 29.78 / 33.79 |
| Real-3DQA debiased EM / EM-R, VRS | 15.76 / 19.73, 11.03 | 15.69 / 19.60, 10.83 |
| Beacon3D GPT-score ScanNet / 3RScan / MultiScan | 44.91 / 41.30 / 25.52 | 45.04 / 41.54 / 25.58 |

The tables above are the default path. Wall clock on 2 Blackwell GPUs (`run_infer.sh`, all
benchmarks, 15127 questions + 141 captioned scenes): **28.0 min -> 3.2 min (8.7×)**; per task
e.g. Scan2Cap 504 -> 85 s, ScanQA 324 -> 26 s, Beacon3D-ScanNet 84 -> 7 s. Peak GPU memory
12 GB per process (default 5.5 GB).

## Dependencies: `deps/`

All checkpoints, data, raw scans, the JRE and the environment a run reads are reached through
the gitignored `deps/` (hard links or copies, or symlinks only into storage owned by this
baseline), and `env.sh` points there, so deleting a shared copy elsewhere does not break
re-runs. The venv's interpreter is a copy too (`pyvenv.cfg` home -> `deps/uv-python-3.10.22`).

## Files

* `build_env.sh`, `download.sh`, `run_infer.sh`, `score.sh` — the steps above; paths from `env.sh`
* `infer.py` — inference, default path and `--fast` (see above)
* `score.py` — Scan2Cap / ScanQA (repo scorers), SQA3D / Real-3DQA (Real-3DQA matching and
  official scripts), Beacon3D QA (official evaluator)
* `prepare_scans.py` — Beacon3D 3RScan / MultiScan scenes in LL3DA's `scannet_data` layout
* `compare_preds.py`, `check_fast_fp32.py` — agreement and exactness checks of `--fast`
* `xr/` — XR-QA / XR-SceneCaption: `prepare_xr.py` (items, region points + normals), `xr_data.py`
  (scene assembly), `score_xr.py`, `finetune_xr.py`, and their `.sh` runners
* `real3dqa/`, `beacon3d/Beacon3D/` — the official evaluators, vendored unmodified
