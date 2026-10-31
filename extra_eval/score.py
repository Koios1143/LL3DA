#!/usr/bin/env python3
"""Score ``infer.py`` outputs. CPU only; METEOR needs Java.

* Scan2Cap: the repo's ``eval_utils/evaluate_densecap.py`` -- per ScanRefer val object
  (``prepare_corpus``), the caption of the kept proposal (objectness + NMS) with the highest box
  IoU above the threshold, ``"sos eos"`` if none, scored with the repo's ``utils/capeval``
  against the ``sos ... eos`` corpus; IoU 0.25 and 0.5 (the paper's Table 5 uses 0.5).
* ScanQA val: the repo's ``eval_utils/evaluate_qa.py`` scoring (``utils/capeval`` BLEU /
  CIDEr / METEOR / ROUGE-L, no PTB tokenizer; the paper's Table 5 numbers), plus EM / EM-R.
* SQA3D test, Real-3DQA, ScanQA EM / EM-R: Real-3DQA's official ``answer_match`` /
  ``clean_answer`` (exact / either-substring match after cleaning; vendored in ``real3dqa/``);
  SQA3D also per question type. Real-3DQA runs the official ``evaluate_debiased_exact_match.py``
  and ``evaluate_rotation_robustness.py``. An empty generation is replaced by ``<empty>``
  first (otherwise the substring match counts it as correct).
* Beacon3D QA (ScanNet, 3RScan, MultiScan): the official ``evaluate_qa.py`` (vendored in
  ``beacon3d/Beacon3D``, unmodified) with the GPT judge ``--qa_model``; no grounding run.

Writes ``<result_dir>/<bench>/scores.json`` (+ predictions) and ``<result_dir>/summary.json``.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "real3dqa"))
REAL3DQA_SPLITS = ["debiased_test", "rotation_0", "rotation_90", "rotation_180", "rotation_270"]
EXPECTED = {"scanqa": 4675, "sqa3d_test": 3519, "real3dqa_debiased_test": 1485, "real3dqa_rotation_0": 750,
            "real3dqa_rotation_90": 750, "real3dqa_rotation_180": 750, "real3dqa_rotation_270": 750,
            "beacon3d_scannet_qa": 1155, "beacon3d_3rscan_qa": 816, "beacon3d_multiscan_qa": 477,
            "scan2cap": 141}
EMPTY = "<empty>"
SQA_TYPES = ["what", "is", "how", "can", "which", "others"]
BEACON_TAGS = ["class", "appearance", "geometry", "spatial", "existence", "functionality"]


def load(pred_dir, task):
    rows = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, task, "predictions_chunk_*.json"))):
        for r in json.load(open(f)):
            rows.setdefault(r["qid"], r)
    if not rows:
        return None
    assert len(rows) == EXPECTED[task], f"{task}: {len(rows)} predictions, expected {EXPECTED[task]}"
    return rows


def capeval(corpus, candidates):
    from eval_utils.evaluate_densecap import score_captions
    _, _, m = score_captions(corpus, candidates)
    # score_captions reports BLEU-4 / CIDEr / ROUGE / METEOR; BLEU-1 from capeval directly
    import utils.capeval.bleu.bleu as capbleu
    b = capbleu.Bleu(4).compute_score(corpus, candidates)[0]
    return {"CIDEr": 100 * m["CiDEr"], "BLEU-1": 100 * b[0], "BLEU-4": 100 * m["BLEU-4"],
            "METEOR": 100 * m["METEOR"], "ROUGE-L": 100 * m["Rouge"]}


def score_scan2cap(rows, args):
    from eval_utils.evaluate_densecap import prepare_corpus
    scanrefer = json.load(open(os.path.join(args.scanrefer_dir, "ScanRefer_filtered_val.json")))
    corpus, object_id_to_name = prepare_corpus(scanrefer)
    res = {}
    for thr in (0.25, 0.5):
        cap, iou = OrderedDict(), {}
        for r in rows.values():
            for p in r["proposals"]:
                if p["iou"] <= thr:  # evaluate_densecap: match_box_ious > test_min_iou
                    continue
                name = object_id_to_name[f"{r['scene_id']}|{p['obj_id']}"]
                key = f"{r['scene_id']}|{p['obj_id']}|{name}"
                if p["iou"] > iou.get(key, 0.0):
                    iou[key], cap[key] = p["iou"], [p["caption"]]
        matched = len(corpus.keys() & cap.keys())
        for k in corpus.keys() - cap.keys():
            cap[k] = ["sos eos"]
        cands = OrderedDict((k, v) for k, v in sorted(cap.items()) if not k.endswith("unknown"))
        m = capeval(OrderedDict((k, corpus[k]) for k in cands), cands)
        res.update({f"{k}@{thr}": v for k, v in m.items()})
        res[f"matched@{thr}"] = f"{matched}/{len(corpus)}"
    return res


def em_scores(rows, by_type=False):
    import evaluate_debiased_exact_match as dem
    em, emr, per = [], [], {}
    for r in rows:
        pred = dem.clean_answer(r["pred"])
        pred = pred if pred.strip() else EMPTY
        e, er = dem.answer_match(pred, [dem.clean_answer(a) for a in r["answers"]])
        em.append(e)
        emr.append(er)
        if by_type:
            per.setdefault(SQA_TYPES[r["sqa_type"]], []).append((e, er))
    out = {"EM": 100 * float(np.mean(em)), "EM-R": 100 * float(np.mean(emr)),
           "empty_preds": sum(not dem.clean_answer(r["pred"]).strip() for r in rows)}
    for t in SQA_TYPES if by_type else []:
        if t in per:
            out[f"EM_{t}"] = 100 * float(np.mean([x[0] for x in per[t]]))
            out[f"EM-R_{t}"] = 100 * float(np.mean([x[1] for x in per[t]]))
    return out


def score_scanqa(rows):
    # evaluate_qa.py: corpus key = question_id-question, candidates = [answer]
    corpus = OrderedDict(("-".join((r["qid"], r["question"])), r["answers"]) for r in rows.values())
    cands = OrderedDict(("-".join((r["qid"], r["question"])), [r["pred"]]) for r in rows.values())
    return {**capeval(corpus, cands), **em_scores(rows.values())}


def score_real3dqa(pred_dir, out):
    import evaluate_debiased_exact_match as dem
    import evaluate_rotation_robustness as rot
    res, files = {}, {}
    for split in REAL3DQA_SPLITS:
        rows = load(pred_dir, f"real3dqa_{split}")
        if rows is None:
            return None
        files[split] = out / f"pred_{split}.json"
        json.dump([{"question_id": r["qid"], "scene_id": r["scene_id"], "response_gt": r["answers"],
                    "response_pred": r["pred"] if dem.clean_answer(r["pred"]).strip() else EMPTY,
                    "prompt": r["question"]} for r in rows.values()], open(files[split], "w"), indent=1)
        res[split] = dem.evaluate_em(str(files[split]))
        res[split]["empty_preds"] = sum(not dem.clean_answer(r["pred"]).strip() for r in rows.values())
    res["rotation"] = rot.quick_consistency_score([str(files[s]) for s in REAL3DQA_SPLITS[1:]])
    log = []
    for cmd in [[sys.executable, str(HERE / "real3dqa" / "evaluate_debiased_exact_match.py"),
                 str(files["debiased_test"]), "--name", "LL3DA"],
                [sys.executable, str(HERE / "real3dqa" / "evaluate_rotation_robustness.py")]
                + [str(files[s]) for s in REAL3DQA_SPLITS[1:]] + ["--name", "LL3DA"]]:
        log.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    (out / "official_scripts.txt").write_text("\n".join(log).replace(str(REPO) + "/", ""))
    return res


def beacon_metrics(pq):
    """Beacon3D QA metrics from ``evaluate_qa.py``'s processed items."""
    gpt = lambda xs: (float(np.mean(xs)) - 1) / 4 * 100  # noqa: E731
    sc = [it["score"] for it in pq]
    m = {"case": gpt(sc),
         "object": 100 * float(np.mean([all(x >= 4 for x in sc[i:i + 3]) for i in range(0, len(sc), 3)])),
         "EM": 100 * float(np.mean([it["em"] for it in pq])),
         "EM-R": 100 * float(np.mean([it["em_r"] for it in pq]))}
    for t in BEACON_TAGS:
        xs = [it["score"] for it in pq if it["tag"].lower() == t]
        m[t] = gpt(xs) if xs else None
    return m


def score_beacon3d(rows, domain, out, args):
    beacon = (HERE / "beacon3d" / "Beacon3D").resolve()
    src = json.load(open(beacon / "data" / domain / f"{domain}_qa.json"))
    infer = [{"question_id": s["question_id"], "scene_id": s["scene_id"], "question": s["question"],
              "pred": rows[i]["pred"], "answers": s["answers"]} for i, s in enumerate(src)]
    json.dump(infer, open(out / "infer_qa.json", "w"), indent=1)
    processed = out / "infer_qa_processed.json"
    processed.unlink(missing_ok=True)  # evaluate_qa.py silently reuses an existing one
    cmd = [sys.executable, "evaluate_qa.py", "--infer", str(out / "infer_qa.json"), "--domain", domain,
           "--model", args.qa_model, "--output", str(processed)]
    p = subprocess.run(cmd, cwd=beacon, capture_output=True, text=True)
    (out / "score_qa.txt").write_text(p.stdout.replace(str(REPO) + "/", ""))
    if p.returncode != 0:
        raise SystemExit(f"{domain}: evaluate_qa.py failed\n{p.stderr[-3000:]}")
    m = beacon_metrics(json.load(open(processed)))
    official = float(p.stdout.split("GPT-Score =")[1].split()[0])
    assert abs(official - m["case"]) < 0.01, (official, m)
    return {"qa_judge": args.qa_model, **m}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="infer.py --output_dir")
    ap.add_argument("--result_dir", default=str(HERE / "results"))
    ap.add_argument("--benches", default="scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d")
    ap.add_argument("--scanrefer_dir", default=os.environ.get("SCANREFER_DIR"))
    ap.add_argument("--qa_model", default="gpt-4o-mini-2024-07-18",
                    help="Beacon3D GPT judge (upstream default gpt-4o-2024-11-20)")
    args = ap.parse_args()
    pred_dir = os.path.abspath(args.pred_dir)
    result_dir = Path(args.result_dir).resolve()
    os.chdir(REPO)
    summary_file = result_dir / "summary.json"
    summary = json.load(open(summary_file)) if summary_file.exists() else {}
    for bench in args.benches.split(","):
        jobs = ([(f"beacon3d_{d}", f"beacon3d_{d}_qa") for d in ["scannet", "3rscan", "multiscan"]]
                if bench == "beacon3d" else [(bench, bench)])
        for name, task in jobs:
            out = result_dir / name
            out.mkdir(parents=True, exist_ok=True)
            if name == "real3dqa":
                res = score_real3dqa(pred_dir, out)
            else:
                rows = load(pred_dir, task)
                if rows is None:
                    res = None
                else:
                    json.dump(list(rows.values()), open(out / "preds.json", "w"), indent=1)
                    if name == "scan2cap":
                        res = score_scan2cap(rows, args)
                    elif name == "scanqa":
                        res = score_scanqa(rows)
                    elif name == "sqa3d_test":
                        res = em_scores(rows.values(), by_type=True)
                    else:
                        res = score_beacon3d(rows, name[len("beacon3d_"):], out, args)
            if res is None:
                print(f"[{name}] no predictions in {pred_dir}, skipped")
                continue
            res = {"pred_dir": os.path.relpath(pred_dir, REPO), **res}
            json.dump(res, open(out / "scores.json", "w"), indent=1)
            summary[name] = res
            print(f"[{name}]", json.dumps(res, indent=1), flush=True)
        if bench == "beacon3d" and all(n in summary for n, _ in jobs):
            # Beacon3D defines no cross-domain score: all questions pooled, and the mean of the domains
            pooled = [it for n, _ in jobs for it in json.load(open(result_dir / n / "infer_qa_processed.json"))]
            per = [summary[n] for n, _ in jobs]
            overall = {"qa_judge": args.qa_model, "pooled": beacon_metrics(pooled),
                       "mean": {k: float(np.mean([p[k] for p in per if p.get(k) is not None]))
                                for k in beacon_metrics(pooled)}}
            (result_dir / "beacon3d_overall").mkdir(parents=True, exist_ok=True)
            json.dump(overall, open(result_dir / "beacon3d_overall" / "scores.json", "w"), indent=1)
            summary["beacon3d_overall"] = overall
    json.dump(summary, open(summary_file, "w"), indent=1)


if __name__ == "__main__":
    main()
