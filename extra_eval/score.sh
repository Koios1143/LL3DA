#!/bin/bash
# Score extra_eval/run_infer.sh outputs (extra_eval/score.py). CPU only; METEOR needs Java,
# Beacon3D QA needs OPENAI_API_KEY.
#   bash extra_eval/score.sh
#   OUT         run_infer.sh output dir                   (default eval_outputs/extra_eval)
#   BENCHES     comma list of scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d   (default all)
#   QA_MODEL    Beacon3D QA judge                         (default gpt-4o-mini-2024-07-18)
#   RESULT_DIR  where results go                          (default extra_eval/results)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
PY="${LL3DA_ENV:-.venv}/bin/python"
$PY extra_eval/score.py --pred_dir "${OUT:-eval_outputs/extra_eval}" --result_dir "${RESULT_DIR:-extra_eval/results}" \
    --benches "${BENCHES:-scan2cap,scanqa,sqa3d_test,real3dqa,beacon3d}" --scanrefer_dir "${SCANREFER_DIR:-}" \
    --qa_model "${QA_MODEL:-gpt-4o-mini-2024-07-18}" || exit 1
echo SCORE_DONE
