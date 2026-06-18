# UPert

Perturbation-response prediction with graph neural networks.

## Overview

- data loading and preprocessing utilities for LINCS L1000 and OmniPath
- UPert / GAT training entry points
- baseline evaluation code for DeepCOP and GSNN
- figure generation scripts

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Small metadata files are in the repo. Expression data comes from LINCS L1000.

### Download

Get the Level 3 `.gctx` files from [clue.io/data](https://clue.io/data) (requires free account) and put them in `data/cmap/`:
- `level3_beta_ctl_n188708x12328.gctx`
- `level3_beta_trt_cp_n1805898x12328.gctx`

### Convert

```bash
python src/data/data_preprocess.py
```

This creates `data/cmap/level3_beta_ctl_n188708x12328.h5` and
`data/cmap/level3_beta_trt_cp_n1805898x12328.h5`.

## Repository Layout

- `src/data/`   — data preprocessing, loading, and graph export helpers
- `src/model/`  — UPert / GAT and DeepCOP model definitions
- `src/run/`    — main training and evaluation entry points
- `src/plot/`   — figure generation scripts
- `src/utils/`  — experiment utilities and helpers

## Main Entry Points

### UPert / GAT runs

- `src/run/train_models.py` — unified entrypoint, use `--variant` to select model:

| `--variant` | description |
|---|---|
| `base`         | original model + CF loss |
| `control`      | control-context + CF loss + sparse GAT |
| `hybrid`       | hybrid-context + CF loss |
| `attention`    | context-attention + CF loss |
| `control_nocf` | control-context without CF loss |

### Baselines

- `src/run/train_deepcop.py`
- `src/run/train_gsnn_eval.py`

## Running Scripts

Before running any script, set up `PYTHONPATH` so the flat imports work:

```bash
export PYTHONPATH=src/data:src/model:src/utils:src/run
```

When data lives on a separate volume, point `--root` to the data root:

```bash
export ROOT=/path/to/data
```

## Reproducing Thesis Results

All results were produced on an internal server.  Below are the exact commands, first with the legacy entrypoints, then with the current unified entrypoint.

### UPert no-CF

Control-context model **without** the counterfactual constraint.

```bash
# --- Legacy ---
python src/run/train_gat_run_no_cf_drug_loss_control_context.py \
  --root $ROOT \
  --cell_line ALL \
  --epochs 50 \
  --batch_size 32 \
  --attention_layers 4 \
  --predict_uncertainty \
  --pcc_lambda 5.0 \
  --save_meta_json outputs/no_cf_0523/ugat_no_cf_uncertainty_sparse.meta.json \
  --save_eval_npz outputs/no_cf_0523/ugat_no_cf_uncertainty_sparse.eval.npz

# --- Current ---
python src/run/train_models.py --variant control_nocf \
  --root $ROOT \
  --cell_line ALL \
  --epochs 50 \
  --batch_size 32 \
  --attention_layers 4 \
  --predict_uncertainty \
  --pcc_lambda 5.0 \
  --save_meta_json outputs/no_cf_0523/ugat_no_cf_uncertainty_sparse.meta.json \
  --save_eval_npz outputs/no_cf_0523/ugat_no_cf_uncertainty_sparse.eval.npz
```

### UPert with CF

Control-context model **with** the counterfactual drug constraint (cf_lambda = 5.0).

```bash
# --- Legacy ---
python src/run/train_gat_run_cf_drug_loss_control_context.py \
  --root $ROOT \
  --cell_line ALL \
  --epochs 50 \
  --batch_size 32 \
  --attention_layers 4 \
  --cf_lambda 5.0 \
  --cf_margin 0.2 \
  --predict_uncertainty \
  --pcc_lambda 5.0 \
  --save_meta_json outputs/with_cf_gat_0524/cagnn_control_context.meta.json \
  --save_eval_npz outputs/with_cf_gat_0524/cagnn_control_context.eval.npz

# --- Current ---
python src/run/train_models.py --variant control \
  --root $ROOT \
  --cell_line ALL \
  --epochs 50 \
  --batch_size 32 \
  --attention_layers 4 \
  --cf_lambda 5.0 \
  --cf_margin 0.2 \
  --predict_uncertainty \
  --pcc_lambda 5.0 \
  --save_meta_json outputs/with_cf_gat_0524/cagnn_control_context.meta.json \
  --save_eval_npz outputs/with_cf_gat_0524/cagnn_control_context.eval.npz
```

### DeepCOP baseline

```bash
python src/run/train_deepcop.py \
  --root $ROOT \
  --cell_line ALL \
  --epochs 10 \
  --batch_size 256 \
  --lr 5e-4 \
  --split_modes warm,cold_target_pattern,cold_cell \
  --predict_uncertainty \
  --pcc_lambda 5.0 \
  --save_json outputs/deepcop_uncertainty_0524/deepcop_uncertainty.json
```

### GSNN baseline

```bash
python src/run/train_gsnn_eval.py \
  --root $ROOT \
  --cell_line ALL \
  --use_landmark_genes \
  --split_modes warm,cold_target_pattern,cold_cell \
  --epochs 50 \
  --batch_size 32 \
  --channels 8 \
  --layers 3 \
  --dropout 0.1 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --norm none \
  --node_mlp_hidden 64 \
  --save_json outputs/gsnn_0524/gsnn_results.json \
  --save_pred_prefix outputs/gsnn_0524/gsnn_results.pred
```

## Notes

- Data folders, baseline repositories, and generated outputs are intentionally excluded from this repository.
- Some auxiliary plotting and checking scripts are kept for reproducibility even if they are not part of the final thesis text.
