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
# XR-SceneCaption SPICE, same setup as just3d: SPICE 1.0 jars in deps/spice, JDK 17 in deps/jdk17
if [ -f deps/spice/spice-1.0.jar ] && [ -x deps/jdk17/bin/java ]; then
    export ALIGNED_SPICE_DIR="$PWD/deps/spice" ALIGNED_SPICE_JAVA="$PWD/deps/jdk17/bin/java"
    export ALIGNED_SPICE_CACHE="${ALIGNED_SPICE_CACHE:-$PWD/deps/spice_cache}"
    export ALIGNED_SPICE_SHARDS="${ALIGNED_SPICE_SHARDS:-6}" ALIGNED_SPICE_HEAP="${ALIGNED_SPICE_HEAP:-6G}"
else
    echo "[warn] deps/spice or deps/jdk17 missing: XR-SceneCaption scored without SPICE" >&2
fi
PY="${LL3DA_ENV:-deps/venv}/bin/python"
"$PY" extra_eval/aligned/score_aligned.py --only "${ONLY:-sqa3d_test,scanqa,scan2cap,xr}" || exit 1
echo SCORE_ALIGNED_DONE
