#!/usr/bin/env python3
"""XR-QA / XR-SceneCaption (LSceneLLM) items and per-region points for LL3DA.

An XR scene is a set of SceneVerse HM3D regions (XR-QA ``qa_pairs[].region_id``,
XR-SceneCaption ``room_id``). Each region's point cloud (240k points, xyz + rgb 0-255,
self-centred) is cached once with normals, which LL3DA's input needs (``use_normal``; on
ScanNet they are the mesh's vertex normals, SceneVerse has points only): PCA over 16 nearest
neighbours, oriented towards the region's centre. ``xr_data.py`` assembles a scene from its
regions at load time, in one of two placements (LSceneLLM's ``room_center`` or the real
HM3D layout).

Writes ``<out_dir>/regions/<scan>_<sub>.npz`` (xyz float32, rgb uint8, normals float16),
``<out_dir>/xr_<split>_items.json`` (one record per question: scene alias, task, question,
LL3DA instruction text, answers, LSceneLLM episode key) and ``xr_<split>_scene_map.json``.
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def qa_items(xr_dir, split):
    """LSceneLLM's HD_Hm3dQADataset order; key = its episode_id."""
    items = []
    for scan, eps in json.load(open(os.path.join(xr_dir, "XR-QA", f"{split}.json"))).items():
        for ep in eps:
            for qa in ep["qa_pairs"]:
                items.append(dict(task="qa", scan=scan, regions=qa["region_id"].split("-"),
                                  question=qa["question"], answers=[qa["answer"]], type=qa["type"],
                                  key="HM3D#{}#{}#{}".format(scan, qa["region_id"], qa["question"])))
    return items


def cap_items(xr_dir, split):
    """``instruction`` keeps the dataset's ``### human: ... ### assistant:`` form, which is the
    form of LL3DA's own scene-description instructions (``unified_3dllm_scene_description``)."""
    items = []
    for i, x in enumerate(json.load(open(os.path.join(xr_dir, "XR-SceneCaption", f"{split}.json")))):
        q = re.sub(r"^\s*###\s*human:\s*|\s*###\s*assistant:\s*$", "", x["question"]).strip()
        items.append(dict(task="cap", scan=x["scan_id"], regions=x["room_id"].split("-"), question=q,
                          instruction=x["question"].strip(), answers=x["answers"],
                          key="HM3D#{}#{}#{}#{}".format(x["scan_id"], x["room_id"], q, i)))
    return items


def estimate_normals(xyz, k=16):
    _, nbr = cKDTree(xyz).query(xyz, k=k)
    n = np.empty_like(xyz)
    for s in range(0, len(xyz), 50000):
        p = xyz[nbr[s:s + 50000]]                       # m x k x 3
        p = p - p.mean(1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("mki,mkj->mij", p, p))
        n[s:s + 50000] = vec[:, :, 0]                   # smallest eigenvalue
    centre = np.array([*np.median(xyz[:, :2], 0), np.percentile(xyz[:, 2], 50)])
    flip = ((centre - xyz) * n).sum(1) < 0  # point the normal towards the room interior
    n[flip] *= -1
    return n


def region_job(job):
    reg, scan_data, out = job
    if out.exists():
        return reg
    xyz, rgb, _ = torch.load(Path(scan_data) / "pcd_with_global_alignment" / f"{reg}.pth", weights_only=False)
    xyz = np.asarray(xyz, np.float64)
    normals = estimate_normals(xyz)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, xyz=xyz.astype(np.float32), rgb=np.clip(np.round(rgb), 0, 255).astype(np.uint8),
             normals=normals.astype(np.float16))
    os.replace(tmp, out)
    return reg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True, help="dir with XR-QA/ and XR-SceneCaption/")
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "regions").mkdir(parents=True, exist_ok=True)

    regions = set()
    for split in args.splits:
        items, scenes = [], {}
        for task, its in (("qa", qa_items(args.xr_dir, split)), ("cap", cap_items(args.xr_dir, split))):
            prefix = {"qa": "xq", "cap": "xc"}[task] + ("t" if split == "train" else "")
            local = {}
            for it in its:
                k = (it["scan"], tuple(it["regions"]))
                if k not in local:
                    local[k] = f"{prefix}{len(local):04d}"
                    scenes[local[k]] = dict(scan=k[0], regions=list(k[1]))
                it["scene_id"] = local[k]
                regions.update(f"{k[0]}_{r}" for r in k[1])
            items += its
        for i, it in enumerate(items):
            it["idx"] = i
        json.dump(items, open(out / f"xr_{split}_items.json", "w"))
        json.dump(scenes, open(out / f"xr_{split}_scene_map.json", "w"), indent=0)
        print(f"{split}: {len(items)} items, {len(scenes)} scenes", flush=True)

    jobs = [(r, args.scan_data, out / "regions" / f"{r}.npz") for r in sorted(regions)]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(region_job, jobs, chunksize=1)):
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"regions {i + 1}/{len(jobs)}", flush=True)
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
