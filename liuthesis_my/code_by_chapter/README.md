# Code By Chapter

This folder organizes the project code by thesis chapter without moving the original source files.

The goal is to make it easy to find:
- data preprocessing code
- model and training code
- figure generation code
- table sources
- external baseline implementations

Each chapter index below points to the original files that belong to that chapter.

## Index

- `01_introduction_theory.md`
- `02_data.md`
- `03_methodology.md`
- `04_results.md`
- `05_discussion_case_studies.md`
- `06_appendix_supporting_material.md`
- `07_external_baselines.md`

## Main Source Roots

- `src/` holds the main project code for preprocessing, training, evaluation, and plotting.
- `liuthesis_my/figures/result_generators/` holds thesis specific figure generation scripts.
- `liuthesis_my/*.tex` holds the final thesis text and most final tables.
- `results/`, `deepcop_res/`, `gsnn_res/`, and `gat_res/` hold exported experiment outputs.
- `DeepCOP/`, `GSNN/`, and `XPert/` hold external or reference baseline implementations.
