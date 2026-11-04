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
bash extra_eval/aligned/run_score_aligned.sh   # CPU; cross-baseline re-scoring, see "Aligned scoring"
```

XR-QA / XR-SceneCaption have their own scripts (see their section).

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
SPLITS="train val" bash extra_eval/xr/prepare_xr.sh   # per-region points + mesh normals -> $XR_CACHE
bash extra_eval/xr/run_xr.sh                          # zero-shot, both placements, lscenellm protocol
XR_PROTOCOL=ll3da bash extra_eval/xr/run_xr.sh        # zero-shot, LL3DA's own protocol
XR_PROTOCOL=ll3da bash extra_eval/xr/run_xr_finetune.sh   # finetune (headline), one placement per visible GPU
bash extra_eval/xr/run_xr_finetune.sh                 # finetune with the lscenellm prompt
```

Each writes predictions + `*.scores.json` under `eval_outputs/`; `CAP_DECODE=beam` gives the
`*_beam` captions. Paths: `XR_DIR`, `XR_SCAN_DATA`, `XR_CACHE`, `HM3D_GLB`, `NORMALS_ENV`
(`env.sh.example`).

**Inputs.** LL3DA needs normals (`use_normal`); SceneVerse regions are points only.
`prepare_xr.py` takes them from the HM3D v0.2 meshes (`NORMALS=glb`, the default): each region
is moved into the glb frame with its registered offset (`region_offsets.json`), and every point
gets the face normal of its closest triangle (Open3D `RaycastingScene`, geometry only). Points
farther than 5 cm from the mesh, or on a degenerate face, fall back to PCA normals (16
neighbours, oriented to the region's centre; `NORMALS=pca` uses these everywhere, as our first
run did). A region whose median distance to the mesh is above 25 cm aborts (wrong frame or
offset); above 1 cm it is logged. The mesh agrees with the points: median distance 0.003 cm,
PCA fallback for 0.14% of the points, and only 5 regions have a median above 1 cm (one at
24.7 cm, `00894-HY1NcmCgn3n_sub005`, where 66% of its points fall back to PCA). These numbers
come from `normals_stats.json`, which covers 1,503 of the 2,035 regions (the rest were written
by a run that stopped before saving its statistics). `check_glb_normals.py` compares them with PCA and checks the
orientation: floor normals point up, ceiling normals down. Two placements of the regions:
**rc** as LSceneLLM's loader (region + `room_center`) and **glb** the real house layout. 40k
points per scene, seeded per scene.

**Protocol (`--xr_protocol`).** The paper "re-implements LL3DA keeping all other settings the
same as ours", i.e. inside LSceneLLM's framework. Only that reproduces its zero-shot row
(`lscenellm`, the default): the scene is centred and scaled into the unit sphere (`pc_norm`),
and the question goes into LSceneLLM's `hd_scene_qa` prompt as written. LL3DA's own setup
(`ll3da`: metric coordinates, its ScanQA prompt, lower-cased question) scores zero-shot QA ~9
CIDEr higher (rc: 34.12 vs 25.26). In the first (PCA-normal) run, unit coords + ScanQA prompt gave 33.50
and metric + `hd_scene_qa` 27.00, so the prompt accounts for most of the gap.
Captions use LL3DA's chat decoding (`predict_chat`: top-k 50 / top-p 0.95 sampling, seeded
per scene); with beam search (`CAP_DECODE=beam`) they are ~130–180 words and lower in CIDEr.

CIDEr / ROUGE-L / METEOR, mesh normals; the PCA-normal run in brackets (QA CIDEr):

| Setting | XR-QA | XR-SceneCaption |
| --- | --- | --- |
| zero-shot, `lscenellm`, rc | 25.26 / 12.08 / 5.77 (24.52) | 0.44 / 10.80 / 5.77 |
| zero-shot, `lscenellm`, glb | 25.08 / 12.03 / 5.75 (24.81) | 0.36 / 10.95 / 5.74 |
| zero-shot, `ll3da`, rc | 34.12 / 16.05 / 7.23 (34.40) | 0.36 / 10.48 / 6.73 |
| zero-shot, `ll3da`, glb | 33.22 / 15.76 / 7.23 (33.68) | 0.33 / 10.75 / 6.25 |
| zero-shot, paper | 24.78 / 12.66 / 5.31 | 0.12 / 8.71 / 5.14 |
| finetuned, `ll3da` prompt, rc, epoch 3 | **115.04 / 36.80 / 18.82** | 1.99 / 16.72 / 11.92 |
| finetuned, `ll3da` prompt, glb, epoch 3 | 114.11 / 36.72 / 18.72 | 2.09 / 16.21 / 11.44 |
| finetuned, `lscenellm` prompt, rc, epoch 3 | 118.04 / 37.56 / 19.03 (114.38) | 1.78 / 17.35 / 12.18 |
| finetuned, `lscenellm` prompt, glb, epoch 3 | 113.72 / 36.51 / 18.58 (114.85) | 1.48 / 17.01 / 11.93 |
| finetuned, paper | 112.80 / 36.94 / 18.68 | 3.22 / 20.95 / 13.49 |

- **Finetuning** (`finetune_xr.py`) follows the repo's generalist training: detector and
  OPT-1.3B frozen, the rest trained (111.0 M) with `CaptionNet`'s loss, AdamW (wd 0.1),
  per-iteration cosine to 1e-6, no warm-up, gradient clipping 0.1, batch 32. Data: XR data
  only (the paper adds LL3DA's own data): XR-QA train + XR-SceneCaption train, one item per
  answer (19,440), a fresh 40k-point sample per item; 3 epochs (~20 min each on one GPU,
  including evaluation); training and evaluation use the same protocol (`XR_PROTOCOL`). LL3DA's own prompt
  (`ll3da`) is the headline; `lscenellm` as the paper. lr 1e-4 (the repo's); LSceneLLM's own
  1e-5 reaches only 102.14 / 33.80 / 17.47 on QA after 3 epochs (`finetune_rc_lr1e-5/`; PCA
  normals 102.91). A preempted run resumes after its last finished epoch (`resume.pt`).
- **QA matches the paper with either prompt; captions stay below it** (best: `lscenellm` rc
  epoch 2, 2.30 / 17.98 / 12.84). The missing LL3DA training data is a likely cause; not tested.
- **Mesh vs PCA normals:** zero-shot moves by < 1 CIDEr. After finetuning, rc gains 3.7 and glb
  loses 1.1 QA CIDEr. These are single runs, so the differences are within seed-to-seed spread;
  the normals do not change the conclusions.
- Placement barely matters. Scenes with > 100 SceneVerse objects score like the rest.

Predictions and scores (all, ≤ 100 / > 100 objects; `raw` and `ll3da`, which lower-cases the
caption references as `evaluate_scene_description.py` does; the tables use `raw`):
[`results/xr/`](results/xr) (`zeroshot/` = `lscenellm`, `zeroshot_ll3da_protocol/`,
`finetune_ll3daprompt_{rc,glb}/epoch<k>/`, `finetune_{rc,glb}/epoch<k>/` and
`finetune_rc_lr1e-5/` with the `lscenellm` prompt; `*_beam` = beam-search captions), summary in
[`results/xr/summary.json`](results/xr/summary.json). The PCA-normal run is kept in
[`results/xr/pca_normals/`](results/xr/pca_normals).

## Aligned scoring

`extra_eval/aligned/` re-scores the saved predictions (no inference) with one text
normalisation shared by all baselines, so these numbers are comparable across baselines; the
paper-protocol scores above stay in their own result files.

- **Scan2Cap:** PTB tokenizer + `sos … eos` on both sides, references untruncated
  (`deps/aligned/scan2cap_val_corpus.json`); LL3DA's own proposal matching / IoU as above.
- **ScanQA, XR-QA:** only the model's special tokens stripped (no truncation), `clean_answer`
  on both sides, then PTB, for CIDEr / BLEU / METEOR / ROUGE-L. EM / EM-R: `clean_answer` on
  both sides, exact match over all references checked before the substring (refined) match.
- **SQA3D:** EM / EM-R the same way (equal to the numbers above). Real-3DQA and Beacon3D unchanged.
- **XR-SceneCaption:** special tokens stripped, no truncation, PTB on both sides (no clean, no
  `sos/eos`). XR is scored per item (2,994 QA / 283 captions, no episode dedup) and also keeps
  a `raw` block (LSceneLLM's protocol, no PTB).
- In zero-shot with `zeroshot_ll3da_protocol`, 37–53 of the 283 captions and 60–81 of the 2,994
  QA predictions run on into a self-generated `### human:` turn; they are kept (no truncation).

**SPICE (XR-SceneCaption)**, computed as in just3d: SPICE 1.0 (CoreNLP 3.6.0 + nashorn) on JDK 17;
captions are split into sentences of at most 40 words (whitespace-free or initial-like run-ons over
120 tokens are re-chunked so CoreNLP's parser does not run out of memory); score = mean per-item
All-F x100, an empty prediction scores 0. Needs `deps/spice/` (`spice-1.0.jar` + `lib/`) and
`deps/jdk17/` (gitignored); without them the script warns and skips SPICE. Headline: zero-shot 5.14,
finetune 11.52 (the `"spice"` block in each XR-SceneCaption JSON; parse cache in `deps/spice_cache/`).

```bash
bash extra_eval/aligned/run_score_aligned.sh          # CPU only; Java from deps/jre11 unless JAVA_HOME
ONLY=scanqa,xr bash extra_eval/aligned/run_score_aligned.sh   # subset (sqa3d_test,scanqa,scan2cap,xr)
```

Outputs: [`results/aligned/`](results/aligned) (`<bench>__<variant>.json`, all XR runs incl.
PCA normals and `*_beam`) and [`summary.json`](results/aligned/summary.json).

| Benchmark | Metric | Aligned | Paper protocol (above) |
|---|---|---:|---:|
| Scan2Cap@0.5 | CIDEr | 61.39 | 63.19 |
| ScanQA val | CIDEr / EM / EM-R | 75.12 / 19.06 / 34.40 | 75.10 / 15.87 / 34.40 |
| SQA3D test | EM / EM-R | 29.89 / 33.96 | 29.89 / 33.96 |
| XR-QA zero-shot, `ll3da`, rc | CIDEr (clean_ptb) / EM | 71.75 / 13.36 | 34.12 / – |
| XR-QA zero-shot, `lscenellm`, rc | CIDEr (clean_ptb) / EM | 54.20 / 9.02 | 25.26 / – |
| XR-QA finetuned, `ll3da` prompt, rc, ep. 3 | CIDEr (clean_ptb) / EM | 125.92 / 22.08 | 115.04 / – |
| XR-SceneCaption zero-shot, `ll3da`, rc | CIDEr (ptb) | 0.46 | 0.36 |
| XR-SceneCaption finetuned, `ll3da` prompt, rc, ep. 3 | CIDEr (ptb) | 2.39 | 1.99 |

ScanQA EM rises because exact match is now tried against every reference before the substring
match.

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
* `xr/` — XR-QA / XR-SceneCaption: `prepare_xr.py` (items, region points + mesh normals),
  `check_glb_normals.py` (normal orientation check), `xr_data.py` (scene assembly), `score_xr.py`,
  `finetune_xr.py`, and their `.sh` runners
* `aligned/` — aligned re-scoring: `aligned_eval.py` (shared scorer), `score_aligned.py`,
  `run_score_aligned.sh`
* `real3dqa/`, `beacon3d/Beacon3D/` — the official evaluators, vendored unmodified
