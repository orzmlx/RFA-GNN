# External Baselines And Reference Repositories

## DeepCOP

Reference implementation and data:

- `DeepCOP/`
- `src/train_deepcop.py`
- `src/deepcop_target/train_deepcop_target.py`
- `src/deepcop.py`

Use this group when you need:

- the original dense baseline logic
- reference preprocessing for DeepCOP style inputs
- original helper code

## GSNN

Reference implementation and library code:

- `GSNN/`
- `src/train_gsnn_eval.py`

Use this group when you need:

- the GSNN model definition
- sparse graph layer implementation
- GSNN explanation utilities

## XPert

Reference repository used mainly for related work, context, and figure inspiration:

- `XPert/`

This includes:

- `XPert/train_xpert.py`
- `XPert/evaluation_metrics/`
- `XPert/reproducing/`

## Notes

- These folders are not the main thesis source tree.
- They are kept separate because they contain third party or reference implementations.
- For the thesis itself, the main reproducible pipeline is centered in `src/`, `results/`, and `liuthesis_my/`.
