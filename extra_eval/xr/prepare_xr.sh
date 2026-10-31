#!/bin/bash
# XR-QA / XR-SceneCaption items + per-region points with normals -> $XR_CACHE (CPU only;
# regions are cached, so re-runs only rewrite the item lists).
#   bash extra_eval/xr/prepare_xr.sh
#   SPLITS="train val" bash extra_eval/xr/prepare_xr.sh
# Paths come from env.sh (see env.sh.example): LL3DA_ENV, XR_DIR, XR_SCAN_DATA, XR_CACHE.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
"${LL3DA_ENV:-.venv}/bin/python" extra_eval/xr/prepare_xr.py --xr_dir "${XR_DIR:?}" --scan_data "${XR_SCAN_DATA:?}" \
    --out_dir "${XR_CACHE:?}" --splits ${SPLITS:-val} --workers "${WORKERS:-16}"
