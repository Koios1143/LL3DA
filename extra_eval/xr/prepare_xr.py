#!/usr/bin/env python3
"""XR-QA / XR-SceneCaption (LSceneLLM) items and per-region points for LL3DA.

An XR scene is a set of SceneVerse HM3D regions (XR-QA ``qa_pairs[].region_id``,
XR-SceneCaption ``room_id``). Each region's point cloud (240k points, xyz + rgb 0-255,
self-centred) is cached once with normals, which LL3DA's input needs (``use_normal``; on
ScanNet they are the mesh's normals, SceneVerse has points only). ``--normals glb`` (default)
takes them from the HM3D v0.2 mesh the points were sampled from: the normal of the nearest
triangle of ``<glb_dir>/<scan>/<hash>.glb`` (open3d closest point; regions are moved into the
glb frame with ``region_offsets.json``). The points lie on the mesh (median distance 0 cm) and the
mesh's winding is consistent (99% of floor normals up, ceiling normals down). ``--normals pca``
(the first runs) = PCA over 16 nearest neighbours, oriented towards the region's centre: same
axis as the mesh normal for most points, but the wrong side for 5-25% of them (surfaces facing
away from the centre). glb points > 5 cm from the mesh or on degenerate triangles get the PCA
normal; ``<out_dir>/normals_stats.json`` records per region how many. ``xr_data.py`` assembles a scene from its
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


def load_region(scan_data, reg):
    xyz, rgb, _ = torch.load(Path(scan_data) / "pcd_with_global_alignment" / f"{reg}.pth", weights_only=False)
    return np.asarray(xyz, np.float64), rgb


def save_region(out, xyz, rgb, normals):
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, xyz=xyz.astype(np.float32), rgb=np.clip(np.round(rgb), 0, 255).astype(np.uint8),
             normals=normals.astype(np.float16))
    os.replace(tmp, out)


def region_job(job):
    reg, scan_data, out = job
    if not out.exists():
        xyz, rgb = load_region(scan_data, reg)
        save_region(out, xyz, rgb, estimate_normals(xyz))
    return [(reg, None)]


def scan_job(job):
    """glb normals for the regions of one scan (the mesh is loaded once)."""
    scan, regs, scan_data, glb_dir, offsets, out_dir, max_dist = job
    todo = [r for r in regs if not (out_dir / f"{r}.npz").exists()]
    if not todo:
        return [(r, None) for r in regs]
    import open3d as o3d
    import trimesh
    # geometry only: force="mesh" would also merge every texture into one atlas (GBs per scan);
    # dump() applies the same scene-graph transforms, so the frame is the one force="mesh" gives
    scene = trimesh.load(Path(glb_dir) / scan / f"{scan.split('-', 1)[1]}.glb", process=False)
    parts = scene.dump(concatenate=False) if isinstance(scene, trimesh.Scene) else [scene]
    vs, fs, off = [], [], 0
    for m in parts:
        vs.append(np.asarray(m.vertices, np.float64))
        fs.append(np.asarray(m.faces, np.int64) + off)
        off += len(m.vertices)
    del scene, parts
    V, Fc = np.concatenate(vs), np.concatenate(fs)
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(Fc.astype(np.uint32)))
    fnorm = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])  # trimesh's face_normals
    ln = np.linalg.norm(fnorm, axis=1, keepdims=True)
    fnorm = np.where(ln > 1e-12, fnorm / np.maximum(ln, 1e-12), 0.0)
    res = [(r, None) for r in regs if r not in todo]
    for reg in todo:
        xyz, rgb = load_region(scan_data, reg)
        w = xyz + np.asarray(offsets[scan][reg.rsplit("_", 1)[1]]["T"], np.float64)
        hit = rs.compute_closest_points(o3d.core.Tensor(w.astype(np.float32)))
        n = fnorm[hit["primitive_ids"].numpy().astype(np.int64)]
        dist = np.linalg.norm(hit["points"].numpy() - w, axis=1)
        # a frame / offset bug puts the points metres off; a few cm = a slightly off region offset
        assert np.median(dist) < 0.25, f"{reg}: median distance to the glb {np.median(dist):.3f} m (wrong frame/offset?)"
        if np.median(dist) > 0.01:
            print(f"note: {reg} median distance to the glb {np.median(dist) * 100:.1f} cm", flush=True)
        bad = (dist > max_dist) | (np.linalg.norm(n, axis=1) < 0.5)
        if bad.any():
            n[bad] = estimate_normals(xyz)[bad]
        save_region(out_dir / f"{reg}.npz", xyz, rgb, n)
        res.append((reg, dict(n=len(xyz), dist_med_cm=float(np.median(dist) * 100),
                              dist_p99_cm=float(np.percentile(dist, 99) * 100), pca_fallback=int(bad.sum()))))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xr_dir", required=True, help="dir with XR-QA/ and XR-SceneCaption/")
    ap.add_argument("--scan_data", required=True, help="SceneVerse HM3D scan_data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--normals", choices=["glb", "pca"], default="glb")
    ap.add_argument("--glb_dir", help="--normals glb: HM3D v0.2 glb dir (<scan>/<hash>.glb)")
    ap.add_argument("--offsets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "region_offsets.json"),
                    help="--normals glb: region -> glb translation")
    ap.add_argument("--only_scans", nargs="*", help="debug: only the regions of these scans")
    ap.add_argument("--max_dist", type=float, default=0.05, help="--normals glb: farther points get PCA normals (m)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "regions").mkdir(parents=True, exist_ok=True)

    # one cache = one kind of normals (cached regions are never recomputed)
    tag = out / "normals.txt"
    if tag.exists():
        assert tag.read_text().strip() == args.normals, f"{out} holds {tag.read_text().strip()} normals, not {args.normals}"
    elif any((out / "regions").iterdir()):
        assert args.normals == "pca", f"{out}/regions has normals from before normals.txt (PCA): use a new --out_dir"
        tag.write_text("pca\n")
    else:
        tag.write_text(args.normals + "\n")

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

    if args.only_scans:
        regions = {r for r in regions if r.rsplit("_", 1)[0] in set(args.only_scans)}
    if args.normals == "pca":
        fn, jobs = region_job, [(r, args.scan_data, out / "regions" / f"{r}.npz") for r in sorted(regions)]
    else:
        assert args.glb_dir, "--normals glb needs --glb_dir"
        offsets = json.load(open(args.offsets))
        by_scan = {}
        for r in sorted(regions):
            by_scan.setdefault(r.rsplit("_", 1)[0], []).append(r)
        missing = [s for s in by_scan if s not in offsets]
        assert not missing, f"no glb offset for {missing}"
        fn = scan_job
        jobs = [(s, rs, args.scan_data, args.glb_dir, offsets, out / "regions", args.max_dist) for s, rs in by_scan.items()]
    stats_file = out / "normals_stats.json"
    stats = json.load(open(stats_file)) if stats_file.exists() else {}
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(fn, jobs, chunksize=1):
            for reg, st in res:
                done += 1
                if st is not None:
                    stats[reg] = st
                if fn is scan_job or done % 100 == 0 or done == len(regions):
                    print(f"regions {done}/{len(regions)}" + (f" ({reg.rsplit('_', 1)[0]})" if fn is scan_job else ""), flush=True)
    if args.normals == "glb":
        json.dump(dict(sorted(stats.items())), open(stats_file, "w"), indent=0)
        v = list(stats.values())
        if v:
            print(f"glb normals: {len(v)} regions, dist p99 max {max(x['dist_p99_cm'] for x in v):.2f} cm, "
                  f"PCA fallback {sum(x['pca_fallback'] for x in v)} of {sum(x['n'] for x in v)} points, "
              f"median > 1 cm in {sum(x['dist_med_cm'] > 1 for x in v)} regions", flush=True)
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
