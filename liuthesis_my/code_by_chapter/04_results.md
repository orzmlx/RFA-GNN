# Chapter 4: Results

## Thesis Files

- `liuthesis_my/Result.tex`

## Main Evaluation And Export Code

- `src/plot_eval_figures.py`
- `src/plot/plot_eval_figures.py`
- `src/regenerate_fig43_true_pred.py`
- `rewrite_result.py`

## Result Inputs

- `results/`
- `deepcop_res/`
- `gsnn_res/`
- `gat_res/`

These folders contain exported JSON and other result files used by the final plots and result summaries.

## Thesis Specific Figure Generators

All scripts below are in `liuthesis_my/figures/result_generators/`.

### Benchmark and metric figures

- `generate_benchmark_mean_baseline_figures.py`
- `generate_top_gene_precision_figure.py`
- `generate_gene_distribution_figure.py`

### Uncertainty figures

- `generate_uncertainty_figures.py`
- `generate_gene_level_uncertainty_figure.py`
- `generate_top_uncertain_enrichment.py`

### Ablation and interpretation figures

- `generate_hybrid_context_weight_figure.py`
- `generate_attention_drug_heatmaps.py`

### Data efficiency figures

- `generate_data_efficiency_figure.py`

## Final Result Figure Assets

Important final figure files used directly by `Result.tex` include:

- `evaluation_framework.png`
- `gene_expr_true_pred_all_models_by_split.png`
- `xpert_style_gene_distribution.png`
- `uncertainty_summary_placeholder.png`
- `gene_level_uncertainty_warm.png`
- `uncertainty_case_study_placeholder.png`
- `data_efficiency_warm_real.png`

## Tables

Most final result tables are written directly in:

- `liuthesis_my/Result.tex`

This includes:

- benchmark MSE table
- benchmark PCC table
- top gene precision table
- sign accuracy table
- ablation tables
- uncertainty summary table
- case study term tables

## Notes

- Some earlier benchmark figure scripts still exist even when the final thesis now uses only the table version.
- If a figure is no longer shown in the thesis, the generator may still be useful as a reproducibility artifact.
