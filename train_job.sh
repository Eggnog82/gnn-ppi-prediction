#!/bin/bash
#SBATCH --job-name=train-job        # Job name
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --partition=gpu            # Partition to submit to
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=6G                    # Memory per node
#SBATCH --gpus=v100:2                    # Request 1 GPU
#SBATCH --mail-type=ALL

# Load Conda environment setup
echo "Setting up Conda environment..."
eval "$(conda shell.bash hook)"      # Initialize Conda in the script
# conda env create -f ppi_env_working.yml
conda activate ppi_env_working

# Run training script
echo "Running training script..."
python train.py

# Run testing script
echo "Running testing script..."
python test.py
