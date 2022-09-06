#!/bin/bash

#SBATCH --job-name=spdspjt       # Submit a job named "example"
#SBATCH --nodes=1                # Using 1 node
#SBATCH --gpus=1                 # Using 1 GPU
#SBATCH --time=0-00:10:00        # 10 minute timelimit
#SBATCH --mem=16000MB            # Using 16GB memory
#SBATCH --cpus-per-task=8        # Using 8 cpus per task (srun)
#SBATCH --output=log.txt         # Creating log file

echo ${USER}
eval "$(conda shell.bash hook)"
conda activate spds018              # Activate your conda environment

srun python main.py --epochs 7
