#!/usr/bin/env python3
"""Score LL3DA predictions on XR-QA / XR-SceneCaption.

Metrics are LSceneLLM's ``utils/capeval`` (BLEU-4, CIDEr, METEOR, ROUGE-L, x100), which is
the same code as this repository's ``utils/capeval``, with predictions and references keyed
by LSceneLLM's episode id (repeated keys keep the last item, as in its validation). Two text
treatments:

* ``raw``    LSceneLLM's validation: the decoded text against the references as written.
* ``ll3da``  this repository's own evaluators: ``evaluate_qa.py`` scores QA exactly as
  ``raw``; ``evaluate_scene_description.py`` lower-cases the references (LL3DA is trained on
  lower-cased descriptions). LSceneLLM's Table 1 baselines match their own evaluation
  (checked on Chat-Scene).

Also splits by the number of SceneVerse objects in the scene (<= 100 / > 100).

    python extra_eval/xr/score_xr.py --pred_dir <out>/xrcap_rc --items xr_val_items.json --out x.json
"""

import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import utils.capeval.bleu.bleu as capbleu  # noqa: E402
import utils.capeval.cider.cider as capcider  # noqa: E402
import utils.capeval.meteor.meteor as capmeteor  # noqa: E402
import utils.capeval.rouge.rouge as caprouge  # noqa: E402


def score(corpus, cands):
    keys = list(cands)
    gts = OrderedDict((k, corpus[k]) for k in keys)
    res = OrderedDict((k, cands[k]) for k in keys)
    bleu = capbleu.Bleu(4).compute_score(gts, res)
    cider = capcider.Cider().compute_score(gts, res)
    rouge = caprouge.Rouge().compute_score(gts, res)
    meteor = capmeteor.Meteor().compute_score(gts, res)
    return {"n": len(keys), "CIDEr": 100 * cider[0], "BLEU-4": 100 * bleu[0][3],
            "METEOR": 100 * meteor[0], "ROUGE-L": 100 * rouge[0]}


def splits(corpus, cands, n_obj):
    out = {"all": score(corpus, cands)}
    for name, keep in (("le100", lambda n: n <= 100), ("gt100", lambda n: n > 100)):
        sub = {k: v for k, v in cands.items() if n_obj and keep(n_obj[k])}
        if sub:
            out[name] = score(corpus, sub)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, help="infer.py <output_dir>/<task> (predictions_chunk_*.json)")
    ap.add_argument("--items", required=True, help="xr_<split>_items.json of prepare_xr.py")
    ap.add_argument("--scene_objects", default=None,
                    help="SceneVerse scan_data dir; default $XR_SCAN_DATA (for the <=100 / >100 split)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = {}
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "predictions_chunk_*.json"))):
        for r in json.load(open(f)):
            recs[r["idx"]] = r
    task = recs[next(iter(recs))]["task"]
    items = [it for it in json.load(open(args.items)) if it["task"] == task]
    missing = [it["idx"] for it in items if it["idx"] not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(items)} {task} items have no prediction")

    scan_data = args.scene_objects or os.environ.get("XR_SCAN_DATA")
    n_obj_scene = {}
    if scan_data:
        import torch
        for it in items:
            if it["scene_id"] not in n_obj_scene:
                n_obj_scene[it["scene_id"]] = sum(
                    len(torch.load(os.path.join(scan_data, "instance_id_to_label", f"{it['scan']}_{r}.pth"),
                                   weights_only=False)) for r in it["regions"])
    corpus, cands, n_obj = {}, {}, {}
    for it in items:  # LSceneLLM keys by episode id: repeated keys keep the last item
        corpus[it["key"]] = list(it["answers"])
        cands[it["key"]] = [" ".join(recs[it["idx"]]["pred"].split())]
        n_obj[it["key"]] = n_obj_scene.get(it["scene_id"])
    if not scan_data:
        n_obj = {}
    own = corpus if task == "qa" else {k: [a.lower() for a in v] for k, v in corpus.items()}
    out = {"raw": splits(corpus, cands, n_obj), "ll3da": splits(own, cands, n_obj),
           "mean_pred_words": sum(len(v[0].split()) for v in cands.values()) / len(cands)}
    json.dump(out, open(args.out, "w"), indent=1)
    for norm in ("raw", "ll3da"):
        for sp, v in out[norm].items():
            print(f"{Path(args.out).stem:16} {norm:5} {sp:5}", {m: round(x, 2) for m, x in v.items()})
    print(f"{Path(args.out).stem:16} mean_pred_words {out['mean_pred_words']:.1f}")


if __name__ == "__main__":
    main()
