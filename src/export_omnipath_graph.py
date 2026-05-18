import os
import sys
import pandas as pd

root = "/Users/liuxi/Desktop/RFA_GNN"
src = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from data_loader import load_rfa_data, build_combined_gnn

tf_path = os.path.join(root, "data/omnipath/omnipath_tf_regulons.csv")
ppi_path = os.path.join(root, "data/omnipath/omnipath_interactions.csv")
full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
siginfo_path = os.path.join(root, "data/siginfo_beta.txt")
landmark_path = os.path.join(root, "data/landmark_genes.json")
ctl_path = os.path.join(root, "data/cmap/level3_beta_ctl_n188708x12328.gctx")
trt_path = os.path.join(root, "data/cmap/level3_beta_trt_cp_n1805898x12328.gctx")
drug_target_path = os.path.join(root, "data/compound_targets.txt")
fingerprint_path = os.path.join(root, "data/new_morgan_fingerprints.csv")

# We only need the target genes for build_combined_gnn
data = load_rfa_data(
    ctl_path,
    trt_path,
    drug_target_path=drug_target_path,
    landmark_path=landmark_path,
    siginfo_path=siginfo_path,
    fingerprint_path=fingerprint_path,
    use_landmark_genes=True,
    full_gene_path=full_gene_path,
    cell_lines=None,
    max_samples=2, # Fast loading
)

if data is None:
    raise RuntimeError("load_rfa_data returned None")

print("Building graph...")
adj_matrix, node_list, gene2idx, edge_index = build_combined_gnn(
    tf_path=tf_path,
    ppi_path=ppi_path,
    target_genes=data["target_genes"],
    confid_threshold=0.9,
    directed=True,
    omnipath_consensus_only=False, # match default
    omnipath_is_directed_only=False, # match default
    symbol_to_entrez=data.get("symbol_to_entrez"),
)

print(f"Graph built with {len(node_list)} nodes and {edge_index.shape[1]} edges.")

# Now we need to read the original CSVs, apply the same overlap logic to export the fully annotated dataframe
df_tf = pd.read_csv(tf_path)
df_ppi = pd.read_csv(ppi_path)

df_tf['edge_type'] = 'TF_Regulon'
df_ppi['edge_type'] = 'PPI'

df_all = pd.concat([df_tf, df_ppi], ignore_axis=True)
df_all = df_all.dropna(subset=['source_genesymbol', 'target_genesymbol'])

# Filter by target genes in the graph
target_genes_set = set(node_list)
df_filtered = df_all[
    df_all['source_genesymbol'].isin(target_genes_set) & 
    df_all['target_genesymbol'].isin(target_genes_set)
]

out_path = os.path.join(root, "data", "omnipath_preprocessed_graph.csv")
df_filtered.to_csv(out_path, index=False)
print(f"Exported preprocessed graph data to {out_path}")
