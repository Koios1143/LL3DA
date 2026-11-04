#!/usr/bin/env python3
"""Re-score LL3DA's saved predictions with the common text normalisation (``aligned_eval.py``).

No inference: reads the committed predictions only and writes
``extra_eval/results/aligned/<bench>__<variant>.json`` plus ``summary.json``.

* Scan2Cap (``results/scan2cap/preds.json``, per-scene kept proposals with caption, matched
  ScanNet object and box IoU): ``score.py``'s matching -- per corpus object the kept proposal
  with the highest IoU above the threshold (``> thr``), IoU 0.25 and 0.5; unmatched objects get
  ``""``. References: ``deps/aligned/scan2cap_val_corpus.json`` (keys ``scene|objid``, 2068).
  Mode ``s2c``.
* ScanQA val (``results/scanqa/preds.json``): all saved answers; ``clean_ptb``, ``raw`` (+ EM).
* SQA3D test (``results/sqa3d_test/preds.json``): EM / EM-R only, checked against the existing
  ``results/sqa3d_test/scores.json`` (must agree within 0.1).
* XR-QA / XR-SceneCaption (every ``results/xr/**/xr{qa,cap}_*.jsonl``): references saved with
  each prediction; scored per ITEM (keyed by ``idx``: all 2994 XR-QA items, all 283 captions),
  never deduplicated by episode key (unlike ``score_xr.py`` / LSceneLLM, which keep the last of
  13 repeated QA keys -> 2981); ``n_unique_episodes`` is recorded. QA ``clean_ptb``, ``raw``;
  captions ``ptb``, ``raw``.

Special tokens: every prediction was decoded with ``skip_special_tokens=True`` (OPT-1.3B:
``</s>`` is BOS/EOS, ``<pad>``, ``<unk>``), so ``strip_model_tokens`` only guards against them
and removes the ``sos`` / ``eos`` wrapper ``infer.py`` put around Scan2Cap captions (``s2c``
re-adds it). Text the model generated itself (e.g. a continued ``### assistant:`` turn) is kept.

    python extra_eval/aligned/score_aligned.py [--only scan2cap,scanqa,sqa3d_test,xr]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "deps" / "pycocoevalcap-1.2"))  # pycocoevalcap 1.2 (not in the venv)
sys.path.insert(0, str(HERE))
import aligned_eval as ae  # noqa: E402

RES = REPO / "extra_eval" / "results"
OUT = RES / "aligned"
CORPUS = REPO / "deps" / "aligned" / "scan2cap_val_corpus.json"

N_CORPUS, N_SCAN2CAP_SCENES, N_SCANQA, N_SQA3D = 2068, 141, 4675, 3519
N_XR_QA, N_XR_QA_EPISODES, N_XR_CAP = 2994, 2981, 283
N_XR_FILES = 87  # results/xr/**/*.jsonl as of 2026-11-04 (29 QA + 58 caption)
# existing headline: LL3DA's own prompt, finetuned on XR, room-centre placement, epoch 3, chat captions
HEADLINE = {("xr_qa", "finetune_ll3daprompt_rc.epoch3.rc"), ("xr_scenecaption", "finetune_ll3daprompt_rc.epoch3.rc")}
TABLE_EPOCH = 3  # README tables use epoch 3 of every finetune run

SPECIAL = re.compile(r"</s>|<s>|<pad>|<unk>")


def strip_model_tokens(text, sos_eos=False):
    """Drop OPT special tokens (already skipped at decode) and, for Scan2Cap, the sos/eos wrapper."""
    t = " ".join(SPECIAL.sub(" ", text or "").split())
    if sos_eos:
        t = re.sub(r"^sos\b", "", t)
        t = re.sub(r"\beos$", "", t)
        t = t.strip()
    return t


def rel(p):
    return os.path.relpath(p, REPO)


def check(name, got, want):
    if got != want:
        raise SystemExit(f"{name}: got {got}, expected {want}")


def write(summary, bench, variant, payload):
    payload = {"bench": bench, "variant": variant, **payload}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / f"{bench}__{variant}.json", "w"), indent=1)
    summary[f"{bench}__{variant}"] = payload
    print(f"[{bench}__{variant}]", json.dumps(payload.get("metrics"), indent=None), flush=True)


def do_scan2cap(summary):
    src = RES / "scan2cap" / "preds.json"
    rows = json.load(open(src))
    check("scan2cap scenes", len({r["scene_id"] for r in rows}), N_SCAN2CAP_SCENES)
    corpus = json.load(open(CORPUS))
    check("scan2cap corpus", len(corpus), N_CORPUS)
    for thr in (0.25, 0.5):
        best, iou = {}, {}
        for r in rows:
            for p in r["proposals"]:
                if p["iou"] <= thr:  # evaluate_densecap: match_box_ious > test_min_iou
                    continue
                k = f"{r['scene_id']}|{p['obj_id']}"
                if k in corpus and p["iou"] > iou.get(k, 0.0):
                    iou[k], best[k] = p["iou"], strip_model_tokens(p["caption"], sos_eos=True)
        preds = OrderedDict((k, best.get(k, "")) for k in sorted(corpus))
        m = ae.score_captions(corpus, preds, modes=("s2c",))
        write(summary, "scan2cap", f"iou{thr}", {
            "source": rel(src), "n": len(preds), "matched": len(best),
            "protocol": "per corpus object: kept proposal (objectness + NMS) with highest IoU > thr; "
                        "unmatched -> '' (sos eos)",
            "metrics": m})


def do_scanqa(summary):
    src = RES / "scanqa" / "preds.json"
    rows = json.load(open(src))
    check("scanqa rows", len(rows), N_SCANQA)
    check("scanqa qids", len({r["qid"] for r in rows}), N_SCANQA)
    refs = OrderedDict((r["qid"], list(r["answers"])) for r in rows)
    preds = OrderedDict((r["qid"], strip_model_tokens(r["pred"])) for r in rows)
    old = json.load(open(RES / "scanqa" / "scores.json"))
    write(summary, "scanqa", "val", {
        "source": rel(src), "n": len(preds), "metrics": ae.score_qa(refs, preds, modes=("clean_ptb", "raw")),
        "existing_EM": old["EM"], "existing_EM-R": old["EM-R"]})


def do_sqa3d(summary):
    src = RES / "sqa3d_test" / "preds.json"
    rows = json.load(open(src))
    check("sqa3d rows", len(rows), N_SQA3D)
    check("sqa3d qids", len({r["qid"] for r in rows}), N_SQA3D)
    refs = {r["qid"]: list(r["answers"]) for r in rows}
    preds = {r["qid"]: strip_model_tokens(r["pred"]) for r in rows}
    em = ae.em_scores(preds, refs)
    old = json.load(open(RES / "sqa3d_test" / "scores.json"))
    diff = abs(em["EM"] - old["EM"])
    write(summary, "sqa3d", "test", {"source": rel(src), "n": len(preds), "metrics": {"em": em},
                                     "existing_EM": old["EM"], "existing_EM-R": old["EM-R"]})
    if diff > 0.1:
        raise SystemExit(f"sqa3d EM {em['EM']:.2f} differs from the existing {old['EM']:.2f} by > 0.1")


def do_xr(summary):
    files = sorted(glob.glob(str(RES / "xr" / "**" / "xr*_*.jsonl"), recursive=True))
    check("XR prediction files", len(files), N_XR_FILES)
    for f in files:
        relp = Path(f).relative_to(RES / "xr")
        m = re.fullmatch(r"xr(qa|cap)_((?:rc|glb)(?:_beam)?)\.jsonl", relp.name)
        if not m:
            raise SystemExit(f"unexpected XR file {f}")
        task, tail = m.groups()
        variant = ".".join(list(relp.parent.parts) + [tail])
        rows = [json.loads(line) for line in open(f)]
        refs, preds = OrderedDict(), OrderedDict()
        for r in rows:  # per item, never deduplicated by episode key
            k = f"{r['idx']}"
            if k in preds:
                raise SystemExit(f"{relp}: duplicate item idx {k}")
            refs[k] = list(r["answers"])
            preds[k] = strip_model_tokens(r["pred"])
        n_episodes = len({r["key"] for r in rows})
        if task == "qa":
            check(f"{relp} items", len(preds), N_XR_QA)
            check(f"{relp} unique episodes", n_episodes, N_XR_QA_EPISODES)
            bench, metrics = "xr_qa", ae.score_qa(refs, preds, modes=("clean_ptb", "raw"))
        else:
            check(f"{relp} rows", len(rows), N_XR_CAP)
            check(f"{relp} unique episodes", n_episodes, N_XR_CAP)
            bench, metrics = "xr_scenecaption", ae.score_captions(refs, preds, modes=("ptb", "raw"))
        ep = re.search(r"epoch(\d+)", variant)
        payload = {"source": rel(f), "n": len(preds), "n_unique_episodes": n_episodes,
                   "empty_preds": sum(not p for p in preds.values()), "metrics": metrics}
        if ep:
            payload["epoch"] = int(ep.group(1))
            payload["existing_table_epoch"] = int(ep.group(1)) == TABLE_EPOCH
        payload["existing_headline"] = (bench, variant) in HEADLINE
        write(summary, bench, variant, payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="sqa3d_test,scanqa,scan2cap,xr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sfile = OUT / "summary.json"
    summary = json.load(open(sfile)) if sfile.exists() else {}
    steps = {"sqa3d_test": do_sqa3d, "scanqa": do_scanqa, "scan2cap": do_scan2cap, "xr": do_xr}
    for name in args.only.split(","):
        steps[name](summary)
        json.dump(OrderedDict(sorted(summary.items())), open(sfile, "w"), indent=1)
    print("ALIGNED_DONE", flush=True)


if __name__ == "__main__":
    main()
