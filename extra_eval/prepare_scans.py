#!/usr/bin/env python3
"""3RScan / MultiScan scenes of Beacon3D -> LL3DA's ``scannet_data`` layout.

LL3DA's ScanNet input (``data/scannet/batch_load_scannet_data.py``) is the scan mesh's
vertices with colour and face-averaged normals (``scannet_utils.compute_normal``), rotated by
ScanNet's ``axis_align_matrix`` (z up, walls along x / y) and randomly subsampled to 50k
points. The same is built here from the raw meshes (both already metric and z-up, the frame of
SceneVerse / Beacon3D):

* 3RScan: ``mesh.refined.v2.obj``, colours sampled from ``mesh.refined_0.png`` at the vertex UVs
  (the vertices of ``labels.instances.annotated.v2.ply``, whose colours are instance colours);
  MultiScan: ``<scene>.ply`` inside the scene zip (vertex colours).
* Axis alignment (no ``axis_align_matrix`` exists for these): yaw from the dominant wall
  direction (area-weighted circular mean of 4 * azimuth over near-horizontal face normals),
  rotated so walls run along x / y; then x / y centred and the floor (1st z percentile) at 0.
* 50k-point subsample seeded by the scene id.

Writes ``<out_dir>/<domain>/<scene>_{aligned_vert,ins_label,sem_label,aligned_bbox}.npy``
(labels / boxes empty: only QA is run on these domains) and ``<out_dir>/<domain>/meta.json``.
"""

import argparse
import io
import json
import os
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
from plyfile import PlyData

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data" / "scannet"))
from scannet_utils import compute_normal  # noqa: E402

MAX_NUM_POINT = 50000  # batch_load_scannet_data.py


def read_ply(fileobj):
    ply = PlyData.read(fileobj)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)
    return xyz, rgb, faces


def read_rscan(scan_dir):
    """3RScan ``mesh.refined.v2.obj`` with per-vertex colour sampled from its texture: the
    labelled ply has the same vertices and faces, but its colours are the instance colours."""
    from PIL import Image
    v, vt, f = [], [], []
    for line in open(os.path.join(scan_dir, "mesh.refined.v2.obj")):
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("vt "):
            vt.append(line.split()[1:3])
        elif line.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    xyz, uv, faces = np.asarray(v, np.float32), np.asarray(vt, np.float32), np.asarray(f, np.int64)
    assert len(uv) == len(xyz)  # vt index == v index in these files
    lab_xyz, _, lab_faces = read_ply(os.path.join(scan_dir, "labels.instances.annotated.v2.ply"))
    assert np.allclose(lab_xyz, xyz, atol=1e-4) and (lab_faces == faces).all()
    tex = np.asarray(Image.open(os.path.join(scan_dir, "mesh.refined_0.png")).convert("RGB"))
    h, w = tex.shape[:2]
    col = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    row = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return xyz, tex[row, col].astype(np.float32), faces


def wall_yaw(xyz, faces):
    """Yaw (rad) that turns the dominant wall direction onto the x / y axes."""
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) + 1e-12
    n = n / area[:, None]
    vert = np.abs(n[:, 2]) < 0.2
    az = np.arctan2(n[vert, 1], n[vert, 0])
    z = np.sum(area[vert] * np.exp(4j * az))
    return -np.angle(z) / 4


def export(xyz, rgb, faces, seed):
    normals = compute_normal(xyz.astype(np.float64), faces).astype(np.float32)
    yaw = wall_yaw(xyz, faces)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz, normals = xyz @ rot.T, normals @ rot.T
    shift = np.array([-(xyz[:, 0].min() + xyz[:, 0].max()) / 2, -(xyz[:, 1].min() + xyz[:, 1].max()) / 2,
                      -np.percentile(xyz[:, 2], 1)], dtype=np.float32)
    xyz = xyz + shift
    verts = np.concatenate([xyz, rgb, normals], 1)
    if len(verts) > MAX_NUM_POINT:
        rng = np.random.RandomState(seed)
        verts = verts[rng.choice(len(verts), MAX_NUM_POINT, replace=False)]
    return verts, {"yaw_deg": float(np.degrees(yaw)), "shift": shift.tolist(), "n_vertices": int(len(xyz))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beacon3d_dir", required=True, help="vendored Beacon3D (data/<domain>/<domain>_qa.json)")
    ap.add_argument("--rscan_dir", required=True, help="3RScan scans/<scan>/labels.instances.annotated.v2.ply")
    ap.add_argument("--multiscan_dir", required=True, help="MultiScan scans/<scene>.zip")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    for domain in ["3rscan", "multiscan"]:
        scenes = sorted({q["scene_id"] for q in json.load(open(
            os.path.join(args.beacon3d_dir, "data", domain, f"{domain}_qa.json")))})
        out = Path(args.out_dir) / domain
        out.mkdir(parents=True, exist_ok=True)
        meta = {}
        for scene in scenes:
            if domain == "3rscan":
                xyz, rgb, faces = read_rscan(os.path.join(args.rscan_dir, scene))
            else:
                with zipfile.ZipFile(os.path.join(args.multiscan_dir, f"{scene}.zip")) as z:
                    xyz, rgb, faces = read_ply(io.BytesIO(z.read(f"{scene}/{scene}.ply")))
            verts, meta[scene] = export(xyz, rgb, faces, zlib.crc32(scene.encode()))
            np.save(out / f"{scene}_aligned_vert.npy", verts)
            np.save(out / f"{scene}_ins_label.npy", np.zeros(len(verts), dtype=np.int64))
            np.save(out / f"{scene}_sem_label.npy", np.zeros(len(verts), dtype=np.int64))
            np.save(out / f"{scene}_aligned_bbox.npy", np.zeros((0, 8), dtype=np.float32))
            print(domain, scene, meta[scene], flush=True)
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
