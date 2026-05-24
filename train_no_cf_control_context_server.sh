#!/usr/bin/env bash
set -euo pipefail

# Optional environment variables:
# ROOT=/path/to/RFA_GNN
# PYTHON_BIN=python
# CUDA_VISIBLE_DEVICES=0
# EPOCHS=50
# BATCH_SIZE=32
# CELL_LINE=ALL

ROOT="${ROOT:-/local/data1/liume102/rfa}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CELL_LINE="${CELL_LINE:-ALL}"
TEST_FRAC="${TEST_FRAC:-0.2}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.2}"
CELL_DROPOUT_RATE="${CELL_DROPOUT_RATE:-0.3}"
ATTENTION_LAYERS="${ATTENTION_LAYERS:-4}"
PCC_LAMBDA="${PCC_LAMBDA:-5.0}"
CF_MARGIN="${CF_MARGIN:-0.1}"

RUN_NAME="${RUN_NAME:-gat_no_cf_control_context}"
OUT_DIR="${OUT_DIR:-$ROOT/gat_res/${RUN_NAME}}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT"

"$PYTHON_BIN" src/train_gat_run_no_cf_drug_loss_control_context.py \
  --root "$ROOT" \
  --cell_line "$CELL_LINE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --test_frac "$TEST_FRAC" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --dropout "$DROPOUT" \
  --cell_dropout_rate "$CELL_DROPOUT_RATE" \
  --attention_layers "$ATTENTION_LAYERS" \
  --pcc_lambda "$PCC_LAMBDA" \
  --cf_margin "$CF_MARGIN" \
  --save_gat_weights "$OUT_DIR/model.weights.h5" \
  --save_meta_json "$OUT_DIR/meta.json" \
  --save_eval_npz "$OUT_DIR/eval.npz"
