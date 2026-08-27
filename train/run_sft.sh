#!/usr/bin/env bash
# Arm 1 — SFT only. LoRA fine-tune on the verified polars examples.
#
#   bash train/run_sft.sh                          # full run  (~1 epoch)
#   bash train/run_sft.sh 20 smoke                 # quick smoke test
#   bash train/run_sft.sh 700 broad25 mlx-broad25  # a specific dataset
#
# Args: $1 = iterations (default 700)
#       $2 = run name   (default sft-v1)
#       $3 = data dir under data/ (default mlx)
#       $4 = base model (default the stock bf16 model; pass a fused
#            CPT model here to run arm 3, CPT -> SFT)
#
# Why these settings:
#   --fine-tune-type lora  train ~0.3% of the weights, not all of them. Cheap,
#                          and the result is a small shareable file.
#   --mask-prompt          grade only the answer half. Without it the model also
#                          learns to write questions, and the loss curve looks
#                          fine while quality drops.
#   --model ...-bf16       full precision. 4-bit affine costs 10 points on this
#                          task (NOTES F27); starting from it would handicap
#                          every arm.
#   --batch-size 4         2,850 examples / 4 = ~712 steps per epoch.
#   --learning-rate 1e-4   standard for LoRA. Full fine-tuning wants ~100x less.
#   --max-seq-length 512   our prompts are ~200 tokens; larger just wastes memory.
#   --steps-per-eval 100   watch validation loss, not just training loss —
#                          training loss always falls, that tells you nothing.
set -euo pipefail

ITERS="${1:-700}"
NAME="${2:-sft-v1}"
DATA="data/${3:-mlx}"
MODEL="${4:-mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16}"
ADAPTER="adapters/${NAME}"

cd "$(dirname "$0")/.."
mkdir -p "$ADAPTER"

echo "model    : $MODEL"
echo "data     : $DATA  ($(wc -l < $DATA/train.jsonl) train / $(wc -l < $DATA/valid.jsonl) valid)"
echo "iters    : $ITERS"
echo "adapter  : $ADAPTER"
echo

.venv-train/bin/python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 16 \
  --batch-size 4 \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --max-seq-length 512 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --val-batches 20 \
  --save-every 200 \
  --adapter-path "$ADAPTER" \
  2>&1 | tee "$ADAPTER/train.log"

echo
echo "done. score it with:"
echo "  .venv-train/bin/python -m evals.run_eval \\"
echo "      --model mlx:$MODEL@$ADAPTER --prompt v1 --note '$NAME'"
