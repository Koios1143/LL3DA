#!/bin/bash
# Finetune LL3DA on XR-QA + XR-SceneCaption train (see finetune_xr.py), one placement per
# visible GPU in parallel, evaluating on val after every epoch; then score every epoch (CPU).
#   bash extra_eval/xr/run_xr_finetune.sh                 # rc on GPU 0, glb on GPU 1
#   PLACEMENTS=rc bash extra_eval/xr/run_xr_finetune.sh
# Needs SPLITS="train val" bash extra_eval/xr/prepare_xr.sh.
# Knobs: PLACEMENTS ("rc glb"), OUT (default eval_outputs/xr_ft_<timestamp>; one subdir per
#   placement), EPOCHS (3), LR (1e-4, cosine to 1e-6), BATCH_SIZE (32), MICRO_BATCH (8),
#   CAP_DECODE ("chat beam"), WORKERS (8 data loaders per run), XR_PROTOCOL
#   (lscenellm | ll3da, see infer.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${LL3DA_ENV:-.venv}/bin/python"
OUT="${OUT:-eval_outputs/xr_ft_$(date +%Y%m%d_%H%M%S)}"
gpu_list="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
PLS=(${PLACEMENTS:-rc glb})
pids=()
for i in "${!PLS[@]}"; do
    pl=${PLS[$i]}
    mkdir -p "$OUT/$pl"
    CUDA_VISIBLE_DEVICES=${GPULIST[$((i % ${#GPULIST[@]}))]} $PY extra_eval/xr/finetune_xr.py --placement "$pl" \
        --ckpt "$LL3DA_HF/ll3da-opt-1.3b.pth" --vocab "$OPT_PATH" --qformer_vocab "$BERT_EMB" \
        --xr_cache "${XR_CACHE:?}" --xr_scan_data "${XR_SCAN_DATA:?}" --out_dir "$OUT/$pl" \
        --epochs "${EPOCHS:-3}" --lr "${LR:-1e-4}" --batch_size "${BATCH_SIZE:-32}" \
        --micro_batch "${MICRO_BATCH:-8}" --cap_decode ${CAP_DECODE:-chat beam} --workers "${WORKERS:-8}" \
        --xr_protocol "${XR_PROTOCOL:-lscenellm}" \
        > "$OUT/$pl/train.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for pl in "${PLS[@]}"; do
    tail -n 2 "$OUT/$pl/train.log"
    for d in "$OUT/$pl"/epoch*/*/; do
        $PY extra_eval/xr/score_xr.py --pred_dir "$d" --items "$XR_CACHE/xr_val_items.json" \
            --out "${d%/}.scores.json" || rc=1
    done
done
echo "XR_FT_EXIT=$rc out=$OUT"
