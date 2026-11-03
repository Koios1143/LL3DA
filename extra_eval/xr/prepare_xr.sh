#!/bin/bash
# XR-QA / XR-SceneCaption items + per-region points with normals -> $XR_CACHE (CPU only;
# regions are cached, so re-runs only rewrite the item lists).
#   bash extra_eval/xr/prepare_xr.sh
#   SPLITS="train val" bash extra_eval/xr/prepare_xr.sh
#   NORMALS=pca XR_CACHE=<other dir> bash extra_eval/xr/prepare_xr.sh   # the first runs' PCA normals
# Paths come from env.sh (see env.sh.example): XR_DIR, XR_SCAN_DATA, XR_CACHE, and for
# NORMALS=glb (default) HM3D_GLB + NORMALS_ENV (a venv with open3d + trimesh + torch + scipy);
# for NORMALS=pca LL3DA_ENV.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
NORMALS="${NORMALS:-glb}"
if [ "$NORMALS" = glb ]; then PY="${NORMALS_ENV:?}/bin/python"; EXTRA=(--glb_dir "${HM3D_GLB:?}"); else PY="${LL3DA_ENV:-.venv}/bin/python"; EXTRA=(); fi
"$PY" extra_eval/xr/prepare_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --out_dir "${XR_CACHE:?}" --splits ${SPLITS:-val} --workers "${WORKERS:-16}" --normals "$NORMALS" "${EXTRA[@]}"
