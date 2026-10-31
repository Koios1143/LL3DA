#!/usr/bin/env python3
"""Item-level agreement of two ``infer.py`` output dirs (e.g. default path vs ``--fast``).

For every task present in both: QA tasks -- rate of identical answers over common questions;
``scan2cap`` -- rate of identical captions over proposals kept in both runs (and whether the
kept proposals agree). Writes a json summary.
"""

import argparse
import glob
import json
import os


def load(d, task):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, task, "predictions_chunk_*.json"))):
        for r in json.load(open(f)):
            out.setdefault(r["qid"], r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="reference output dir (default path)")
    ap.add_argument("--b", required=True, help="output dir to compare (--fast)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.a, "*", "predictions_chunk_*.json"))}
                   & {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.b, "*", "predictions_chunk_*.json"))})
    res = {}
    for task in tasks:
        a, b = load(args.a, task), load(args.b, task)
        common = sorted(set(a) & set(b), key=str)
        if not common:
            continue
        if task == "scan2cap":
            pa = {(q, p["prop"]): p["caption"] for q in common for p in a[q]["proposals"]}
            pb = {(q, p["prop"]): p["caption"] for q in common for p in b[q]["proposals"]}
            both = set(pa) & set(pb)
            r = {"scenes": len(common), "proposals_a": len(pa), "proposals_b": len(pb), "proposals_both": len(both),
                 "same_caption": round(100 * sum(pa[k] == pb[k] for k in both) / max(len(both), 1), 2)}
        else:
            r = {"items": len(common), "same_text": round(100 * sum(a[q]["pred"] == b[q]["pred"] for q in common)
                                                      / len(common), 2)}
        res[task] = r
        print(task, r)
    json.dump({"a": os.path.relpath(args.a), "b": os.path.relpath(args.b), "tasks": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
