# UPert Thesis Project

## Overview
This repository contains the code, plotting scripts, and thesis materials for the UPert project on perturbation-response prediction with graph neural networks.

The current codebase includes:
- data loading and preprocessing utilities for LINCS L1000 and OmniPath
- multiple UPert / GAT training entry points
- baseline evaluation code for DeepCOP and GSNN
- figure generation scripts used by the thesis
- the thesis source files in `liuthesis_my/`

## Environment Setup

### Prerequisites
- Python 3.10+
- Conda or `venv`

### Quick Start
1. Set up the environment:

```bash
./setup_env.sh
```

This script creates a Conda environment named `RFA_GNN` when Conda is available. Otherwise it creates a local `venv/`.

2. Activate the environment:

If using Conda:

```bash
conda activate RFA_GNN
```

If using `venv`:

```bash
source venv/bin/activate
```

3. Install any optional dependencies you need for plotting or GSNN experiments if they are not already available in your environment.

## Repository Layout
- `src/data/`: data preprocessing, loading, and graph export helpers
- `src/model/`: UPert / GAT and DeepCOP model definitions
- `src/run/`: main training and evaluation entry points
- `src/plot/`: figure generation scripts for thesis figures
- `src/utils/`: experiment utilities and preprocessing helpers
- `src/study/`: exploratory notebooks and case study notebooks
- `liuthesis_my/`: thesis text, generated figures, and chapter-to-code index files

## Main Entry Points

### UPert / GAT runs
- `src/run/train_gat_run_cf_drug_loss.py`
- `src/run/train_gat_run_cf_drug_loss_control_context.py`
- `src/run/train_gat_run_cf_drug_loss_context_attention.py`
- `src/run/train_gat_run_cf_drug_loss_hybrid_context.py`
- `src/run/train_gat_run_no_cf_drug_loss_control_context.py`

### Baselines
- `src/run/train_deepcop.py`
- `src/run/train_gsnn_eval.py`
- `src/deepcop_target/train_deepcop_target.py`

### Example figure scripts
- `src/plot/regenerate_fig43_true_pred.py`
- `src/plot/plot_eval_figures.py`
- `src/plot/plot_l1000_filter_flow.py`
- `src/plot/plot_omnipath_schema.py`

## Running Scripts
Some scripts still use legacy flat imports such as `from data_loader import ...` or `from base_gnn import ...`.

When running scripts directly from the repository root, it is safest to expose the source subdirectories through `PYTHONPATH`:

```bash
PYTHONPATH=src/data:src/model:src/utils:src/run python src/run/train_gat_run_cf_drug_loss.py --help
```

Likewise for plotting:

```bash
PYTHONPATH=src/data:src/model:src/utils:src/run python src/plot/regenerate_fig43_true_pred.py
```

## Thesis Mapping
The thesis source is stored in `liuthesis_my/`.

To locate which code belongs to which chapter, see:
- `liuthesis_my/code_by_chapter/README.md`
- `liuthesis_my/code_by_chapter/01_introduction_theory.md`
- `liuthesis_my/code_by_chapter/02_data.md`
- `liuthesis_my/code_by_chapter/03_methodology.md`
- `liuthesis_my/code_by_chapter/04_results.md`

## Notes
- Large data folders, baseline repositories, and generated outputs are intentionally ignored by Git in this main repository.
- Some auxiliary plotting and checking scripts are kept for reproducibility even if they are not part of the final thesis text.
