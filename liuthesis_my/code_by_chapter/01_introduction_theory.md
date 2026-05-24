# Chapter 1: Introduction And Theory

## Thesis Files

- `liuthesis_my/introduction.tex`
- `liuthesis_my/theory.tex`

## Figures Used Here

- Static or manually prepared figures under `liuthesis_my/figures/`, such as:
  - `high_throughput.png`
  - `well_plates.png`
  - `gene_network_gnn_1.png`
  - `fully_connect_network.png`
  - `graph.png`
  - `graph2adj.png`
  - `attention_mechnisam.png`
  - `multi_head.png`
  - `resNet.png`
  - `FFN.png`
  - `attention_block.png`

## Relevant Plot Scripts

These scripts support concepts introduced in the chapter or generate related schematic figures:

- `src/plot/plot_omnipath_schema.py`
- `src/plot/plot_omnipath_table.py`
- `src/plot/plot_omnipath_graph_978.py`
- `src/plot/plot_drug_omnipath_subgraph_example.py`
- `src/plot/plot_omnipath_tf_path_example.py`

## Notes

- Most theory figures are final image assets rather than generated inside the LaTeX build.
- If a figure was drawn manually or refined outside Python, the image file in `liuthesis_my/figures/` is the final source used by the thesis.
