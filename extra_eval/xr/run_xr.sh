#!/bin/bash
# LL3DA zero-shot on XR-QA / XR-SceneCaption val (both placements), then scoring (CPU).
#   bash extra_eval/xr/run_xr.sh
#   CAP_DECODE=beam TASKS="xrcap_rc xrcap_glb" bash extra_eval/xr/run_xr.sh
# Needs prepare_xr.sh. Knobs: OUT (default eval_outputs/xr_<timestamp>), TASKS
# ("xrqa_rc xrcap_rc xrqa_glb xrcap_glb"), CAP_DECODE (chat | beam), CAP_MAX_LEN, PREFIX_CACHE
# (default 1 here: the KV-cached beam search, see run_infer.sh), as run_infer.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
export OUT="${OUT:-eval_outputs/xr_$(date +%Y%m%d_%H%M%S)}"
export TASKS="${TASKS:-xrqa_rc xrcap_rc xrqa_glb xrcap_glb}" PREFIX_CACHE="${PREFIX_CACHE:-1}"
bash extra_eval/run_infer.sh
rc=0
for t in $TASKS; do
    "${LL3DA_ENV:-.venv}/bin/python" extra_eval/xr/score_xr.py --pred_dir "$OUT/$t" \
        --items "${XR_CACHE:?}/xr_${XR_SPLIT:-val}_items.json" --out "$OUT/$t.scores.json" || rc=1
done
echo "XR_EXIT=$rc out=$OUT"
