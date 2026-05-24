# Chapter 5: Discussion And Case Studies

## Thesis Files

- `liuthesis_my/Discussion.tex`

## Code That Feeds This Chapter

This chapter mainly interprets outputs from the methodology and results pipeline rather than introducing new training code.

The most relevant supporting code is:

- `src/plot_eval_figures.py`
- `liuthesis_my/figures/result_generators/generate_attention_drug_heatmaps.py`
- `liuthesis_my/figures/result_generators/generate_uncertainty_figures.py`
- `liuthesis_my/figures/result_generators/generate_gene_level_uncertainty_figure.py`
- `src/study/plot_eval_figures_case_study.ipynb`
- `src/study/plot_eval_figures_drug_loss_case_study.ipynb`

## Main Inputs

- exported attention results
- exported uncertainty results
- benchmark result JSON files in `results/`

## Tables And Figures

The final tables and narrative discussion are written directly in:

- `liuthesis_my/Discussion.tex`
- `liuthesis_my/Result.tex`

Discussion reuses figures and result tables already generated for the Results chapter, especially:

- uncertainty figures
- drug target contribution tables
- data efficiency figure
- attention case study tables
