from pathlib import Path
import json
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path("/Users/liuxi/Desktop/RFA_GNN")
FIG_DIR = ROOT / "liuthesis_my" / "figures"
NPZ_PATH = ROOT / "results" / "hybrid_context" / "hybrid_context_budget_eval.full.warm.npz"
LANDMARK_PATH = ROOT / "data" / "landmark_genes.json"

TOP_K = 50
LIBRARIES = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
]


def load_symbol_map():
    with LANDMARK_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {str(item["entrez_id"]): str(item["gene_symbol"]) for item in items}


def top_uncertain_genes():
    symbol_map = load_symbol_map()
    z = np.load(NPZ_PATH, allow_pickle=True)
    genes = np.asarray(z["target_genes"], dtype=str)
    y_logvar = np.asarray(z["y_logvar"], dtype=np.float32)
    sigma = np.exp(0.5 * y_logvar)
    mean_sigma = np.mean(sigma, axis=0)
    rank = np.argsort(-mean_sigma)[:TOP_K]
    rows = []
    for idx in rank:
        rows.append(
            {
                "entrez_id": genes[idx],
                "gene_symbol": symbol_map.get(genes[idx], genes[idx]),
                "mean_sigma": float(mean_sigma[idx]),
            }
        )
    return pd.DataFrame(rows)


def enrich_gene_list(gene_list, library):
    add_resp = requests.post(
        "https://maayanlab.cloud/Enrichr/addList",
        files={
            "list": (None, "\n".join(gene_list)),
            "description": (None, "top uncertain warm genes"),
        },
        timeout=30,
    )
    add_resp.raise_for_status()
    user_list_id = add_resp.json()["userListId"]

    enrich_resp = requests.get(
        "https://maayanlab.cloud/Enrichr/enrich",
        params={"userListId": user_list_id, "backgroundType": library},
        timeout=30,
    )
    enrich_resp.raise_for_status()
    rows = enrich_resp.json()[library]
    df = pd.DataFrame(
        rows,
        columns=[
            "rank",
            "term",
            "p_value",
            "z_score",
            "combined_score",
            "overlap_genes",
            "adjusted_p_value",
            "old_p_value",
            "old_adjusted_p_value",
        ],
    )
    df["neg_log10_adj_p"] = -np.log10(np.maximum(df["adjusted_p_value"].astype(float), 1e-300))
    df["overlap_count"] = df["overlap_genes"].apply(len)
    return df


def shorten_term(term, max_len=52):
    term = str(term)
    if len(term) <= max_len:
        return term
    return term[: max_len - 3] + "..."


def plot_library(ax, df, title, color):
    top = df.nsmallest(8, "adjusted_p_value").iloc[::-1].copy()
    top["short_term"] = top["term"].map(shorten_term)
    ax.barh(
        top["short_term"],
        top["neg_log10_adj_p"],
        color=color,
        edgecolor="#7A7A7A",
        linewidth=0.6,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ p)$")
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    out_dir = FIG_DIR / "enrichment_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_df = top_uncertain_genes()
    top_df.to_csv(out_dir / "top_uncertain_genes_warm.csv", index=False)

    dfs = {}
    for library in LIBRARIES:
        df = enrich_gene_list(top_df["gene_symbol"].tolist(), library)
        dfs[library] = df
        df.to_csv(out_dir / f"{library}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=220)
    plot_library(axes[0], dfs["GO_Biological_Process_2023"], "GO Biological Process", "#CFE8C7")
    plot_library(axes[1], dfs["Reactome_2022"], "Reactome", "#D7E6F5")
    fig.suptitle("Pathway enrichment of top uncertain genes (warm)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_uncertain_genes_enrichment_warm.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "top_uncertain_genes_enrichment_warm.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
