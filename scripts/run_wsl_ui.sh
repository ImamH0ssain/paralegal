#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source localenv/bin/activate

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_RUN_ON_SAVE=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.headless true \
  --server.fileWatcherType none \
  --server.runOnSave false
