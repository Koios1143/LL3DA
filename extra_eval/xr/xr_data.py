"""XR scenes as LL3DA input (``ScanNetBaseDataset._get_scan_data`` with ``use_color``,
``use_normal``, ``use_height``, no augmentation).

A scene is the union of its regions (``prepare_xr.py``), each translated by ``placement``:

* ``rc``  LSceneLLM's loader (``SceneVerseDataset.__getitem__``): + ``room_center/<scan>.pth
  [sub]['center']``; only approximate, regions end up ~0.3-0.9 m off from each other.
* ``glb`` + the region's translation into the HM3D v0.2 glb frame (``region_offsets.json``,
  registered in Chat-Scene's ``xr_eval/register_regions.py``), the real layout of the house.

``coords``: ``metric`` keeps metres, as LL3DA's detector was trained on metric ScanNet
scenes; ``unit`` applies LSceneLLM's ``pc_norm`` first (centred, scaled into the unit sphere),
as its loader feeds every model.
"""

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])  # datasets/scannet_base_dataset.py
NUM_POINTS = 40000


class XRScenes:
    def __init__(self, cache_dir, scan_data, placement, coords="metric", offsets=HERE / "region_offsets.json"):
        self.coords = coords
        self.cache = Path(cache_dir)
        self.scan_data = Path(scan_data)
        self.placement = placement
        self.offsets = json.load(open(offsets)) if placement == "glb" else None
        self.room_center = {}
        self.scene_map = {}
        for f in self.cache.glob("xr_*_scene_map.json"):
            self.scene_map.update(json.load(open(f)))

    def translation(self, scan, sub):
        if self.placement == "glb":
            return np.asarray(self.offsets[scan][sub]["T"], np.float32)
        if scan not in self.room_center:
            self.room_center[scan] = torch.load(self.scan_data / "room_center" / f"{scan}.pth", weights_only=False)
        return np.asarray(self.room_center[scan][sub]["center"], np.float32)

    def points(self, scene_id):
        """N x 9: xyz, rgb (0-255), normal."""
        s = self.scene_map[scene_id]
        parts = []
        for sub in s["regions"]:
            r = np.load(self.cache / "regions" / f"{s['scan']}_{sub}.npz")
            parts.append(np.hstack([r["xyz"] + self.translation(s["scan"], sub), r["rgb"].astype(np.float32),
                                    r["normals"].astype(np.float32)]))
        return np.concatenate(parts)

    def inputs(self, scene_id, rng=np.random):
        """``point_clouds`` (40k x 10: xyz, normalised rgb, normal, height) and its xyz range."""
        pc = self.points(scene_id)
        if self.coords == "unit":  # LSceneLLM datasets/SceneVerseDataset.py pc_norm
            xyz = pc[:, :3] - pc[:, :3].mean(0)
            pc[:, :3] = xyz / np.sqrt((xyz ** 2).sum(1)).max()
        pc[:, 3:6] = (pc[:, 3:6] - MEAN_COLOR_RGB) / 256.0
        floor_height = np.percentile(pc[:, 2], 0.99)
        pc = np.concatenate([pc, (pc[:, 2] - floor_height)[:, None]], 1)
        pc = pc[rng.choice(len(pc), NUM_POINTS, replace=len(pc) < NUM_POINTS)]
        return {"point_clouds": pc.astype(np.float32),
                "point_cloud_dims_min": pc[:, :3].min(0).astype(np.float32),
                "point_cloud_dims_max": pc[:, :3].max(0).astype(np.float32)}
