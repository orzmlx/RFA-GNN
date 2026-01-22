# Thesis Proposal: Biologically Informed Graph Neural Networks for Perturbation Effect Prediction

## 1. Background

### 1.1 Problem Statement
A significant challenge in modern medicine is the variability in patient response to treatments, with more than 40% of patients failing to respond to standard therapies for common diseases. This is largely due to the complexity of genomic responses to drugs, driven by a multitude of interactions between genes and proteins within cellular networks. Understanding and predicting these perturbation effects is crucial for precision medicine and drug discovery.

### 1.2 Available Data
To address this problem, we have access to two primary high-quality resources:
*   **Perturbation Data (CMAP/LINCS)**: The Connectivity Map (CMAP) database acts as the **"Training Engine"** for our model. Unlike the static network, CMAP provides dynamic **"Input-Output" pairs** (Drug Target $\to$ Gene Expression Change) derived from real cellular experiments. It serves as the ground truth for supervised learning, allowing the GNN to learn how signals actually propagate in specific cell lines.
*   **Interaction Network (OmniPath)**: A high-confidence, literature-curated resource of signaling pathways, covering over 36,000 causal interactions and ~39% of the human proteome. This serves as our **"Road Map"** or structural backbone.

### 1.3 Specific Research Questions
This project aims to bridge the gap between pure data-driven approaches and knowledge-based approaches by addressing the following questions:
1.  **Model Selection**: Which Graph Neural Network (GNN) architectures are most appropriate for predicting drug perturbation effects when integrating gene expression data with drug target information?
2.  **Generalization**: How accurately can these GNN models predict perturbation effects for new samples (cell lines) and unseen drugs?
3.  **Comparative Analysis**: How much do predictions obtained through biologically informed GNN models differ from and improve upon existing ML models that ignore network structure (e.g., DeepCOP)?

---

## 2. Scientific Context & Literature Review

### 2.1 Previous Approaches
The problem of predicting drug responses has been approached from two distinct angles:

*   **Pure Machine Learning (e.g., DeepCOP)**:
    *   *Method*: DeepCOP (Woo et al., 2020) utilizes deep learning to predict gene regulating effects based on molecular fingerprint descriptors and gene ontology (GO) terms.
    *   *Limitation*: It treats genes as isolated features or simple sets, ignoring the rich, known physical interactions (signaling pathways) that drive these changes. It is "blind" to the biological mechanism.

*   **Pure Network Analysis (e.g., Regulation Flow Analysis - RFA)**:
    *   *Method*: RFA (Roca et al., 2025) models the propagation of signals through the OmniPath network to generate mechanistic hypotheses.
    *   *Limitation*: It relies on static network topology and theoretical propagation rules without leveraging the massive training data available in CMAP to refine its parameters. It is "rigid" and cannot adapt to context-specific data.

### 2.2 What is New in This Study?
This study proposes a **hybrid approach** that integrates the strengths of both worlds.
*   **Novelty**: We are not just using a GNN as a "black box" on a random graph. We are explicitly constructing **Biologically Informed GNNs**.
*   **Innovation**: We aim to use the **OmniPath** network as the structural backbone of the neural network (the adjacency matrix), while using **CMAP** data to learn the non-linear "operators" (feature transformations). This effectively makes the RFA algorithm "learnable," allowing the model to adjust the strength and nature of signal propagation based on real experimental data.

---

## 3. Methodology

### 3.1 Why GNNs?
Biological systems are inherently graph-structured. Proteins do not function in isolation; they form complex signaling cascades.
*   **Reasoning**: Standard neural networks (MLPs) treat input features (genes) as independent or spatially unrelated. GNNs, by design, enforce the prior knowledge that "Gene A influences Gene B" directly into the computation graph.
*   **Selected Model Architecture**: We propose to explore **RFA-GNN (or APPNP-like architectures)**.
    *   *Why*: Standard GCNs often suffer from "over-smoothing" when many layers are stacked, which is detrimental for biological networks where pathways can be long (e.g., 10+ steps). RFA-GNN decouples the "neural transformation" (MLP) from the "graph propagation" (RFA), allowing us to model long-range dependencies without degrading the feature signal.

### 3.2 Alternatives Considered
*   **MLP (Multi-Layer Perceptron)**: The baseline (like DeepCOP). Rejected as the primary model because it ignores topology, though it will be used for comparison.
*   **Bayesian Networks**: feasible for small networks but computationally intractable for genome-scale networks like OmniPath/CMAP.
*   **Static RFA**: Good for explanation but lacks the predictive power of learning from data.

---

## 4. Proposed Implementation Plan

### 4.1 Data Integration
1.  **Network Construction**: Build a signed, directed graph using OmniPath `source` and `target` columns, utilizing `is_stimulation` (+1) and `is_inhibition` (-1) as edge weights.
2.  **Feature Engineering**:
    *   *Input (X)*: Drug targets (one-hot or multi-hot vectors representing perturbed nodes).
    *   *Output (Y)*: Differential gene expression signatures from LINCS (Z-scores).

### 4.2 Model Training
*   **Architecture**: Implement a GNN where the propagation step mimics the RFA algorithm: $H^{(k+1)} = (1-\alpha)H^{(0)} + \alpha A H^{(k)}$.
*   **Learning**: The model will learn the feature transformation matrices (operators) that best map the binary drug target input to the continuous gene expression output via the fixed OmniPath structure.

### 4.3 Evaluation
*   **Metrics**: Pearson correlation between predicted and actual gene expression profiles.
*   **Validation Strategy**: Split data by drugs (predicting effects of novel drugs) and by cell lines (predicting effects in novel contexts).
*   **Benchmark**: Compare performance against the "topology-free" DeepCOP model to quantify the "value of biological structure."

### 4.4 Open Questions & Future Extensions (For Discussion)
This section outlines potential improvements where optimal strategies are yet to be determined. I welcome feedback on these directions:

1.  **Network Densification vs. Quality Trade-off**:
    *   *Problem*: OmniPath is highly curated but sparse. Many genes in the CMAP dataset are "orphans" in the OmniPath network (no incoming edges), making their expression impossible to predict using RFA.
    *   *Proposal*: We could integrate additional data from **STRING** (including high-confidence undirected edges) and **DoRothEA** (including Level B/C confidence TF-target interactions).
    *   *Uncertainty*: Adding these edges increases coverage but introduces noise. It is unclear whether the GNN's ability to learn weights can effectively filter out this noise, or if the "garbage in, garbage out" principle will degrade performance.

2.  **Integration of LncRNA Layer**:
    *   *Problem*: Current networks focus on Protein-Protein and Protein-DNA interactions, ignoring Long Non-coding RNAs (lncRNAs) which are key regulators in many diseases.
    *   *Proposal*: We propose to integrate **LncBook** and **LncRNA2Target** databases to add an lncRNA layer to the graph (`TF -> lncRNA -> Gene`).
    *   *Uncertainty*: Most lncRNA interactions lack directional signs (activation/inhibition). I am unsure about the best strategy to initialize these edge weights (e.g., inferring direction from correlation data vs. initializing as neutral).

3.  **Handling "Undirected" Interactions**:
    *   *Problem*: A significant portion of PPI data (e.g., from STRING) is undirected.
    *   *Proposal*: Treat undirected edges as bidirectional edges in the GNN.
    *   *Uncertainty*: This creates feedback loops. While RFA handles loops mathematically, I am unsure if this will lead to signal explosion or vanishing gradients during GNN training.

4.  **Enhancing CMAP Resolution**:
    *   *Problem*: CMAP relies on measuring only ~978 "Landmark Genes" and inferring the rest. This imputation introduces error.
    *   *Proposal*: We could explore advanced imputation methods (e.g., GANs) or integrate emerging **Single-cell Perturbation (scPerturb)** datasets to improve the quality of our training labels.
    *   *Uncertainty*: Single-cell data is sparse and noisy, which might complicate training compared to the cleaner (albeit lower resolution) bulk data from CMAP.

---

## 5. References
1.  **DeepCOP**: Woo, G., et al. (2020). *DeepCOP: deep learning-based approach to predict gene regulating effects of small molecules*. Bioinformatics.
2.  **OmniPath**: Türei, D., et al. (2016). *OmniPath: guidelines and gateway for literature-curated signaling pathway resources*. Nature Methods.
3.  **RFA**: Roca, C. P., et al. (2025). *Regulation Flow Analysis discovers molecular mechanisms of action from large knowledge databases*. bioRxiv.
4.  **CMAP/LINCS**: Subramanian, A., et al. (2017). *A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles*. Cell.
