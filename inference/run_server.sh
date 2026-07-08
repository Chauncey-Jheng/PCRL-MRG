#!/usr/bin/env bash
# Standalone launcher for the inference server — for running/testing it independently
# of central-control (which instead spawns uvicorn directly via its own Node wrapper,
# see ../../central-control/child-apps/pcrl-mrg/server.js).
set -euo pipefail

PYTHON_BIN=${PCRL_PYTHON:-/home/bjutcv/anaconda3/envs/zcx_llama/bin/python}
HOST=${CHILD_HOST:-0.0.0.0}
PORT=${CHILD_PORT:-8700}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

exec "${PYTHON_BIN}" -m uvicorn server:app --host "${HOST}" --port "${PORT}"
