#!/bin/sh
# Mêmes arguments que LLAMA_ARG_* du compose (embeddings, pooling mean, ctx 2048).
exec llama-server \
  --hf-repo "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0" \
  --embeddings \
  --pooling mean \
  --ctx-size 2048 \
  --threads 2 \
  --host 0.0.0.0 \
  --port "${PORT:-8012}"
