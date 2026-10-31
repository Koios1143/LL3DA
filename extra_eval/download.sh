#!/bin/bash
# Released LL3DA generalist checkpoint + its preprocessed ScanNet point clouds, the OPT-1.3B LLM
# and the BERT embedding of the Q-Former, into the dirs named in env.sh. Idempotent.
#   bash extra_eval/download.sh
#   LL3DA_HF   CH3COOK/LL3DA-weight-release: ll3da-opt-1.3b.pth + scannet_data.zip, unzipped
#   OPT_PATH   facebook/opt-1.3b
#   BERT_EMB   CH3COOK/bert-base-embedding
# Benchmark annotations (SCANREFER_DIR, SCANQA_DIR, SQA3D_TEST, REAL3DQA_DIR) are the public
# ScanRefer / ScanQA / Chat-Scene / Real-3DQA files, see env.sh.example.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
PY="${LL3DA_ENV:-.venv}/bin/python"
HF="${LL3DA_HF:?set LL3DA_HF}"
$PY - "$HF" "${OPT_PATH:?set OPT_PATH}" "${BERT_EMB:?set BERT_EMB}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download as get
hf, opt, bert = sys.argv[1:]
get("CH3COOK/LL3DA-weight-release", local_dir=hf)
get("facebook/opt-1.3b", local_dir=opt, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
get("CH3COOK/bert-base-embedding", local_dir=bert)
PYEOF
if [ ! -d "$HF/scannet_data" ]; then
    (cd "$HF" && unzip -q scannet_data.zip)
fi
ls "$HF" "$OPT_PATH" "$BERT_EMB"
echo DOWNLOAD_DONE
