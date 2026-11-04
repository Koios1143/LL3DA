#!/bin/bash
# Aligned re-scoring of the saved predictions (extra_eval/aligned/score_aligned.py). CPU only,
# no inference; METEOR / PTB need Java (this repo's deps/jre11 unless JAVA_HOME is set).
#   bash extra_eval/aligned/run_score_aligned.sh
#   ONLY   comma list of sqa3d_test,scanqa,scan2cap,xr   (default all)
# Writes extra_eval/results/aligned/<bench>__<variant>.json and summary.json.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f env.sh ] && source env.sh
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export JAVA_HOME="${JAVA_HOME:-deps/jre11}"
export PATH="$(cd "$JAVA_HOME" && pwd)/bin:$PATH"
PY="${LL3DA_ENV:-deps/venv}/bin/python"
"$PY" extra_eval/aligned/score_aligned.py --only "${ONLY:-sqa3d_test,scanqa,scan2cap,xr}" || exit 1
echo SCORE_ALIGNED_DONE
