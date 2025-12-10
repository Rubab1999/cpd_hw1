#!/bin/bash
#SBATCH --job-name=matmul_hw1
#SBATCH --output=matmul_output_%j.txt
#SBATCH --error=matmul_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3

# Create and activate a conda environment (or activate existing one)
# If this is your first time, uncomment the next line to create the environment:
# conda create -n cpd_env python=3.9 -y

# Activate the environment
source activate cpd_env

# Install required packages if not already installed
pip install --user numpy numba

# Run the Python script
python matmul_functions.py

echo "Job completed successfully!"
