#!/bin/bash

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda detected. Setting up Conda environment..."
    
    # Check if environment RFA_GNN exists
    if conda info --envs | grep -q "RFA_GNN"; then
        echo "Environment RFA_GNN already exists."
    else
        echo "Creating Conda environment RFA_GNN..."
        conda create -n RFA_GNN python=3.11 -y
    fi
    
    # Install requirements
    echo "Installing requirements into RFA_GNN..."
    # Using the python executable directly to ensure we install in the right env
    # Note: Adjust path if your conda installation is different
    CONDA_BASE=$(conda info --base)
    $CONDA_BASE/envs/RFA_GNN/bin/pip install -r requirements.txt
    
    echo "Conda environment setup complete."
    echo "To activate: conda activate RFA_GNN"
    
else
    echo "Conda not found. Setting up Python venv..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements..."
        pip install -r requirements.txt
    else
        echo "requirements.txt not found."
    fi

    echo "Venv setup complete."
    echo "To activate: source venv/bin/activate"
fi
