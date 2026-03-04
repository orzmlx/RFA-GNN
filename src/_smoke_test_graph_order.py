import os

from data_loader import build_combined_gnn


def main():
    root = "/Users/liuxi/Desktop/RFA_GNN"
    full_gene_path = os.path.join(root, "data/GSE92742_Broad_LINCS_gene_info.txt")
    landmark_path = os.path.join(root, "data/landmark_genes.json")

    genes_in = ["3", "1", "2", "2", "5"]
    adj, node_list, gene2idx, edge_index = build_combined_gnn(
        tf_path="",
        ppi_path="",
        string_path="",
        full_gene_path=full_gene_path,
        landmark_path=landmark_path,
        landmark_genes=genes_in,
        confid_threshold=0.9,
        directed=False,
    )
    expected = ["3", "1", "2", "5"]
    assert node_list == expected, (node_list[:10], expected)
    assert adj.shape == (len(expected), len(expected)), adj.shape
    print("OK", len(node_list), node_list[:10])


if __name__ == "__main__":
    main()

