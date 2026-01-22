# OmniPath Interaction Dataset Column Descriptions

This file describes the columns present in the OmniPath interaction dataset loaded by `load_omnipath_network`.

## 1. Core Identity Columns (Node Identification)
These columns identify the participating molecules in the interaction.

*   **`source`**: The UniProt ID of the source (upstream) node. UniProt IDs are unique identifiers for proteins (e.g., `P0DP25`).
*   **`target`**: The UniProt ID of the target (downstream) node (e.g., `P48995`).
*   **`source_genesymbol`**: The Gene Symbol of the source node (e.g., `CALM3`). This is the human-readable name commonly used in literature and analysis.
*   **`target_genesymbol`**: The Gene Symbol of the target node (e.g., `TRPC1`).

## 2. Interaction Properties (Edge Attributes)
These columns define the nature and direction of the interaction.

*   **`is_directed`** (Boolean): Indicates if the interaction has a defined direction (`True`) or is undirected (`False`). RFA-GNN typically uses directed edges.
*   **`is_stimulation`** (Boolean): Indicates if the interaction is activating/positive (`True`). The source node upregulates the target.
*   **`is_inhibition`** (Boolean): Indicates if the interaction is inhibiting/negative (`True`). The source node downregulates the target.
*   **`consensus_direction`** (Boolean): Indicates if there is a consensus among multiple databases regarding the direction of the interaction.
*   **`consensus_stimulation`** (Boolean): Consensus indicator for stimulation effects across sources.
*   **`consensus_inhibition`** (Boolean): Consensus indicator for inhibition effects across sources.

## 3. Evidence and Confidence (Quality Control)
These columns provide information about the reliability and origin of the interaction data.

*   **`curation_effort`** (Integer): A metric representing the number of unique resources or databases that have manually curated/verified this interaction. Higher values indicate higher confidence.
*   **`references`**: A list of literature references (e.g., PubMed IDs) that support this interaction (e.g., `TRIP:11290752;...`).
*   **`sources`**: The names of the source databases from which this interaction record was retrieved (e.g., `TRIP`, `SignaLink`, `KEGG`).
*   **`n_sources`** (Integer): The count of different databases that contain this interaction.
*   **`n_references`** (Integer): The count of scientific papers supporting this interaction.

## Usage in RFA-GNN
*   **Graph Structure**: `source_genesymbol` and `target_genesymbol` are used to define the adjacency matrix structure.
*   **Edge Weights**: `is_stimulation` (+1) and `is_inhibition` (-1) determine the sign of the edges in the regulatory network.
