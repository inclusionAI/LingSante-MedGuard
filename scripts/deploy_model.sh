#!/usr/bin/env bash
# Serve a local model through a vLLM OpenAI-compatible endpoint.
set -euo pipefail

if [[ "${1:-}" == "--help" || $# -lt 1 ]]; then
  echo 'Usage: bash deploy_model.sh MODEL [additional vLLM options]'
  echo 'Environment: HOST (127.0.0.1), PORT (8000), TP (1), MODEL_NAME (LingSante MedGuard), VLLM_API_KEY (optional)'
  exit 0
fi

MODEL="$1"
shift
exec "${PYTHON_BIN:-python3}" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${MODEL_NAME:-LingSante MedGuard}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP:-1}" \
  "$@"
