#!/usr/bin/env bash
# Arm 2 — continued pretraining on real polars source (NOTES F51).
#
#   bash train/run_cpt.sh                 # ~1200 iters, ~25 min
#   bash train/run_cpt.sh 30 smoke        # quick smoke test
#   LOWMEM=1 bash train/run_cpt.sh        # if memory is tight (see below)
#
# MEMORY: this peaks at ~11.1 GB (vs 6.5 GB for SFT) because chunks are ~800
# tokens rather than ~200. On an 18 GB machine that is workable only with apps
# closed. LOWMEM=1 adds gradient checkpointing and halves the batch: slower per
# step, roughly half the memory. Watch Memory Pressure, not "Memory Used".
#
# Args: $1 = iterations (default 1200)   $2 = run name (default cpt-v1)
#
# How this differs from run_sft.sh, and why:
#   NO --mask-prompt   CPT is next-token prediction over raw text; every token
#                      carries loss. There is no prompt to mask, and mlx_lm
#                      rejects the flag for text datasets.
#   --learning-rate 5e-5 (SFT used 1e-4). Lower is conventional for continued
#                      pretraining: you are nudging existing knowledge, not
#                      teaching a new response format. NOT swept — if CPT
#                      underperforms, learning rate is an unexcluded
#                      explanation and should be said so in the writeup.
#   --batch-size 2     chunks are ~800 tokens vs SFT's ~200, so fewer per batch
#                      at the same memory.
#   --max-seq-length 1024  matches the chunk size in data_gen/to_mlx_cpt.py.
set -euo pipefail

ITERS="${1:-1200}"
NAME="${2:-cpt-v1}"
MODEL="mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"
ADAPTER="adapters/${NAME}"
DATA="data/mlx-cpt"
if [ "${LOWMEM:-0}" = "1" ]; then BATCH=1; EXTRA="--grad-checkpoint"; else BATCH=2; EXTRA=""; fi

cd "$(dirname "$0")/.."
mkdir -p "$ADAPTER"

echo "model    : $MODEL"
echo "data     : $DATA  ($(wc -l < $DATA/train.jsonl) chunks, ~2.35M tokens)"
echo "iters    : $ITERS   (batch $BATCH, 1 epoch = $(( $(wc -l < $DATA/train.jsonl) / BATCH )) iters)"
echo "lowmem   : ${LOWMEM:-0}"
echo "adapter  : $ADAPTER"
echo

.venv-train/bin/python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size "${BATCH}" \
  --iters "$ITERS" \
  --learning-rate 5e-5 \
  --max-seq-length 1024 \
  --steps-per-report 20 \
  --steps-per-eval 200 \
  --val-batches 20 \
  --save-every 400 \
  --adapter-path "$ADAPTER" ${EXTRA} \
  2>&1 | tee "$ADAPTER/train.log"

echo
echo "next: fuse it, then run SFT on top for arm 3:"
echo "  .venv-train/bin/python -m mlx_lm fuse --model $MODEL \\"
echo "      --adapter-path $ADAPTER --save-path models/${NAME}-fused"
echo "  bash train/run_sft.sh 700 cpt-then-sft mlx-broad25 models/${NAME}-fused"
