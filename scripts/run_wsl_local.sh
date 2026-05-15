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

export HF_HOME="${HF_HOME:-/mnt/f/paralegal/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/f/paralegal/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/f/paralegal/.cache/torch}"
export LOCAL_ACCELERATOR="${LOCAL_ACCELERATOR:-auto}"
export LOCAL_CUDA_MIN_CAPABILITY="${LOCAL_CUDA_MIN_CAPABILITY:-75}"
eval "$(python -m legal_rag.local_device)"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b-instruct}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-900}"
export OLLAMA_STREAM="${OLLAMA_STREAM:-true}"
export OLLAMA_THINK="${OLLAMA_THINK:-false}"
export OLLAMA_MAX_OUTPUT_TOKENS="${OLLAMA_MAX_OUTPUT_TOKENS:-800}"
export OLLAMA_EVIDENCE_LIMIT="${OLLAMA_EVIDENCE_LIMIT:-7}"
export OLLAMA_CHUNK_CHARS="${OLLAMA_CHUNK_CHARS:-550}"
export OLLAMA_NUM_THREAD="${OLLAMA_NUM_THREAD:-8}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-3072}"
export OLLAMA_SUMMARY_CHARS="${OLLAMA_SUMMARY_CHARS:-1800}"
export OLLAMA_FALLBACK_ON_INVALID="${OLLAMA_FALLBACK_ON_INVALID:-true}"
export LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export LOCAL_EMBEDDING_FALLBACK="${LOCAL_EMBEDDING_FALLBACK:-false}"
export LOCAL_RERANK_MODEL="${LOCAL_RERANK_MODEL:-BAAI/bge-reranker-base}"
export LOCAL_RERANK_TOP_N="${LOCAL_RERANK_TOP_N:-10}"
export LOCAL_RERANK_TEXT_CHARS="${LOCAL_RERANK_TEXT_CHARS:-1200}"
export LOCAL_RERANK_WEIGHT="${LOCAL_RERANK_WEIGHT:-0.25}"
export LOCAL_RERANK_PROTECT_TOP="${LOCAL_RERANK_PROTECT_TOP:-2}"
export INGEST_WORKERS="${INGEST_WORKERS:-1}"

python -m legal_rag.cli all \
  --ocr-backend docling \
  --embedding-backend local \
  --reranker-backend local \
  --use-ollama
