"""Compare LL3DA prepare_xr.py PCA normals with normals of the nearest HM3D glb triangle (open3d raycasting)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, trimesh, open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_xr import estimate_normals

scan_data, glb_root, offsets, regs = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4:]
cache = {}
for reg in regs:
    scan, sub = reg.rsplit("_", 1)
    if scan not in offsets or sub not in offsets[scan]:
        print(f"{reg}: no offset (not an XR scan), skipped"); continue
    xyz, rgb, inst = torch.load(Path(scan_data) / "pcd_with_global_alignment" / f"{reg}.pth", weights_only=False)
    xyz = np.asarray(xyz, np.float64)
    w = xyz + np.asarray(offsets[scan][sub]["T"])
    if scan not in cache:
        t0 = time.time()
        m = trimesh.load(Path(glb_root) / scan / f"{scan.split('-')[1]}.glb", force="mesh", process=False)
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
        cache = {scan: (m, sc)}
        print(f"{scan}: {len(m.faces)} faces, load {time.time() - t0:.0f}s", flush=True)
    m, sc = cache[scan]
    t0 = time.time()
    r = sc.compute_closest_points(o3d.core.Tensor(w.astype(np.float32)))
    fid = r["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(r["points"].numpy() - w, axis=1)
    fn = np.asarray(m.face_normals)[fid]
    pca = estimate_normals(xyz)
    cos = (fn * pca).sum(1)
    centre = np.array([*np.median(xyz[:, :2], 0), np.percentile(xyz[:, 2], 50)])
    inward = ((centre - xyz) * fn).sum(1) > 0
    z = xyz[:, 2]
    def orient(n, sel, sign):  # % of horizontal surfaces in sel whose normal points up (sign=1) / down (-1)
        h = sel & (np.abs(fn[:, 2]) > 0.9)
        return np.mean(np.sign(n[h, 2]) == sign) * 100 if h.any() else float("nan"), int(h.sum())
    flo, cei = z < np.percentile(z, 2) + 0.2, z > np.percentile(z, 98) - 0.2
    (fg, nf), (fp, _) = orient(fn, flo, 1), orient(pca, flo, 1)
    (cg, nc), (cp, _) = orient(fn, cei, -1), orient(pca, cei, -1)
    print(f"{reg}: {len(xyz)} pts, closest {time.time() - t0:.1f}s | dist med {np.median(dist)*100:.2f} cm p90 {np.percentile(dist, 90)*100:.2f} cm >5cm {np.mean(dist > .05)*100:.1f}%"
          f" | |cos| med {np.median(np.abs(cos)):.3f} >0.9 {np.mean(np.abs(cos) > .9)*100:.1f}% <0.5 {np.mean(np.abs(cos) < .5)*100:.1f}%"
          f" | sign agree {np.mean(cos > 0)*100:.1f}% | glb inward {inward.mean()*100:.1f}%"
          f" | floor up glb {fg:.1f}% pca {fp:.1f}% (n={nf}) | ceiling down glb {cg:.1f}% pca {cp:.1f}% (n={nc})", flush=True)
