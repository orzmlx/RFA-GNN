# Chapter 3: Methodology

## Thesis Files

- `liuthesis_my/methodology.tex`

## Main Model Code

### Proposed model variants

- `src/base_gnn.py`
- `src/base_gnn_control_context.py`
- `src/base_gnn_context_attention.py`
- `src/base_gnn_hybrid_context.py`
- `src/drug_encoder.py`
- `src/gat_entrypoint_utils.py`

### Training entry points

- `src/train_base_gnn.py`
- `src/train_gat_run_cf_drug_loss.py`
- `src/train_gat_run_cf_drug_loss_control_context.py`
- `src/train_gat_run_cf_drug_loss_context_attention.py`
- `src/train_gat_run_cf_drug_loss_context_attention_no_cellid.py`
- `src/train_gat_run_cf_drug_loss_hybrid_context.py`
- `src/train_gat_run_no_cf_drug_loss_control_context.py`
- `src/train_gat_cf_contrastive.py`
- `src/train_common.py`

## Baseline Training Code Used In Methodology

- `src/train_deepcop.py`
- `src/deepcop_target/train_deepcop_target.py`
- `src/deepcop.py`
- `src/basemodel.py`
- `src/train_gsnn_eval.py`

## Supporting Files

- `src/train_tf_common.py`
- `src/smiles_encoder_tf.py`
- `train_no_cf_control_context_server.sh`

## Method Figures

Many final methodology figures are stored as image assets in `liuthesis_my/figures/`, including:

- `per_node_ctl_input.png`
- `per_node_drug_input.png`
- `cellid_input.png`
- `cell_context.png`
- `base_model_structure.png`
- `proposed_model.png`
- `mean_pcc.png`
- `mean_cell.png`

## Tables

Most methodology tables are written directly in:

- `liuthesis_my/methodology.tex`

This includes:

- model configurations
- parameter count tables
- baseline setup tables
