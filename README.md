# RFA-GNN Project

## Project Description
This project implements the RFA-GNN (Recursive Feature Aggregation Graph Neural Network) model for simulating biological regulatory networks.

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x

### Quick Start

1.  **Run the setup script:**
    ```bash
    ./setup_env.sh
    ```
    *This script will automatically detect Conda and create an environment named `RFA_GNN` if available. Otherwise, it will create a local `venv`.*

2.  **Activate the environment:**
    *   **If using Conda:**
        ```bash
        conda activate RFA_GNN
        ```
    *   **If using venv:**
        ```bash
        source venv/bin/activate
        ```

3.  **Run the implementation:**
    ```bash
    python src/min_implementation.py
    ```

## File Structure
- `src/min_implementation.py`: Minimal implementation of the RFA-GNN model.
- `references/`: Contains relevant research papers.
- `requirements.txt`: Python dependencies.
- `setup_env.sh`: Script to set up the environment.

## Dependencies
- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter
