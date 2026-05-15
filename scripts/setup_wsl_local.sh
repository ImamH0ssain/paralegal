#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-localenv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-local.txt

cat <<'MSG'

WSL local environment is ready.

Recommended one-time Ollama model pull:
  ollama pull qwen3:4b-instruct

Recommended local run:
  source localenv/bin/activate
  export LOCAL_ACCELERATOR=auto
  export LOCAL_CUDA_MIN_CAPABILITY=75
  export HF_HOME=/mnt/f/paralegal/.cache/huggingface
  export TRANSFORMERS_CACHE=/mnt/f/paralegal/.cache/huggingface
  export TORCH_HOME=/mnt/f/paralegal/.cache/torch
  export OLLAMA_MODEL=qwen3:4b-instruct
  export OLLAMA_TIMEOUT=900
  export OLLAMA_STREAM=true
  export OLLAMA_THINK=false
  export OLLAMA_MAX_OUTPUT_TOKENS=800
  export OLLAMA_EVIDENCE_LIMIT=7
  export OLLAMA_CHUNK_CHARS=550
  export OLLAMA_NUM_THREAD=8
  export OLLAMA_NUM_CTX=3072
  export OLLAMA_SUMMARY_CHARS=1800
  export OLLAMA_FALLBACK_ON_INVALID=true
  export LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
  export LOCAL_EMBEDDING_FALLBACK=false
  export LOCAL_RERANK_MODEL=BAAI/bge-reranker-base
  export LOCAL_RERANK_TOP_N=10
  export LOCAL_RERANK_WEIGHT=0.25
  export LOCAL_RERANK_PROTECT_TOP=2
  export INGEST_WORKERS=1
  python -m legal_rag.cli all --ocr-backend docling --embedding-backend local --reranker-backend local --use-ollama

The runner auto-selects CUDA only when PyTorch sees a GPU with compute capability >= sm_75.
Set LOCAL_ACCELERATOR=cpu to force the laptop-safe CPU path, or LOCAL_ACCELERATOR=cuda to force GPU.
If the first run is too slow, use --reranker-backend none or --embedding-backend hashing.
MSG
