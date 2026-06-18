from __future__ import annotations

import os
from pathlib import Path


def project_paths(root: str | Path | None = None) -> dict[str, Path | None]:
    if root is None:
        root = Path(__file__).resolve().parents[3]
    else:
        root = Path(root).resolve()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_tex = root / "Result.tex"
    return {
        "root": root,
        "fig_dir": fig_dir,
        "data_dir": root / "data",
        "outputs_dir": root / "outputs",
        "results_dir": root / "results",
        "result_tex": result_tex if result_tex.exists() else None,
    }
