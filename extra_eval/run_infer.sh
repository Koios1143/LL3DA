#!/bin/bash
# LL3DA inference on Scan2Cap, ScanQA, SQA3D test, Real-3DQA and Beacon3D QA, one process per
# visible GPU (whole scenes per process).
#   bash extra_eval/run_infer.sh                         # the repo's own generation code
#   PREFIX_CACHE=1 bash extra_eval/run_infer.sh          # fast path (infer.py --fast), same outputs
#   TASKS="scanqa sqa3d_test" PREFIX_CACHE=1 bash extra_eval/run_infer.sh   # a subset
# Paths come from env.sh (see env.sh.example):
#   LL3DA_ENV      venv from build_env.sh                         (default .venv)
#   LL3DA_HF       ll3da-opt-1.3b.pth + scannet_data/ (download.sh)
#   OPT_PATH, BERT_EMB, SCANREFER_DIR, SCANQA_DIR, SQA3D_TEST, REAL3DQA_DIR
#   BEACON3D_SCANS prepare_scans.py output for 3RScan / MultiScan (made here if missing,
#                  from RSCAN_RAW_DIR / MULTISCAN_RAW_DIR)
#   OUT            output dir                                      (default eval_outputs/extra_eval)
#   TASKS          space-separated tags                            (default: scan2cap scanqa sqa3d_test
#                  real3dqa_{debiased_test,rotation_0,rotation_90,rotation_180,rotation_270}
#                  beacon3d_{scannet,3rscan,multiscan}_qa)
#   BATCH          default path: items per forward                 (default 8, as scripts/opt-1.3b/eval.*.sh)
#   PREFIX_CACHE   1: infer.py --fast                              (default 0)
#   GEN_BATCH      with PREFIX_CACHE=1: sequences per beam search  (default 32)
#   XR_CACHE, XR_SCAN_DATA, XR_PROTOCOL (lscenellm | ll3da), CAP_DECODE (chat | beam), CAP_MAX_LEN
#                  for xrqa_* / xrcap_* (extra_eval/xr/)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${LL3DA_ENV:-.venv}/bin/python"
OUT="${OUT:-eval_outputs/extra_eval}"
REAL3DQA="real3dqa_debiased_test real3dqa_rotation_0 real3dqa_rotation_90 real3dqa_rotation_180 real3dqa_rotation_270"
TASKS="${TASKS:-scan2cap scanqa sqa3d_test $REAL3DQA beacon3d_scannet_qa beacon3d_3rscan_qa beacon3d_multiscan_qa}"
BEACON3D_SCANS="${BEACON3D_SCANS:-eval_outputs/beacon3d_scans}"
if [[ " $TASKS " == *" beacon3d_3rscan_qa "* || " $TASKS " == *" beacon3d_multiscan_qa "* ]] \
        && [ ! -f "$BEACON3D_SCANS/multiscan/meta.json" ]; then
    $PY extra_eval/prepare_scans.py --beacon3d_dir extra_eval/beacon3d/Beacon3D \
        --rscan_dir "${RSCAN_RAW_DIR:?set RSCAN_RAW_DIR}" --multiscan_dir "${MULTISCAN_RAW_DIR:?set MULTISCAN_RAW_DIR}" \
        --out_dir "$BEACON3D_SCANS" || exit 1
fi
gpu_list="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
FAST=(); [ "${PREFIX_CACHE:-0}" = 1 ] && FAST=(--fast --gen_batch "${GEN_BATCH:-32}")
echo "out=$OUT tasks=[$TASKS] chunks=$CHUNKS prefix_cache=${PREFIX_CACHE:-0}"
mkdir -p "$OUT"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $PY extra_eval/infer.py --tasks "$TASKS" \
        --ckpt "$LL3DA_HF/ll3da-opt-1.3b.pth" --vocab "$OPT_PATH" --qformer_vocab "$BERT_EMB" \
        --scannet_data "$LL3DA_HF/scannet_data" --scanrefer_dir "${SCANREFER_DIR:-}" --scanqa_dir "${SCANQA_DIR:-}" \
        --sqa3d_test "${SQA3D_TEST:-}" --real3dqa_dir "${REAL3DQA_DIR:-}" \
        --beacon3d_dir extra_eval/beacon3d/Beacon3D --beacon3d_scans "$BEACON3D_SCANS" \
        --xr_cache "${XR_CACHE:-}" --xr_scan_data "${XR_SCAN_DATA:-}" \
        --cap_decode "${CAP_DECODE:-chat}" --cap_max_len "${CAP_MAX_LEN:-256}" \
        --xr_protocol "${XR_PROTOCOL:-lscenellm}" \
        --output_dir "$OUT" --num_chunks "$CHUNKS" --chunk_idx "$IDX" --batch "${BATCH:-8}" "${FAST[@]}" \
        > "$OUT/infer_chunk_$IDX.log" 2>&1 &
done
wait
tail -n 2 "$OUT"/infer_chunk_*.log
grep -q INFER_DONE "$OUT/infer_chunk_0.log" && echo INFER_DONE
