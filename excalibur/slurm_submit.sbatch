#!/bin/bash

#SBATCH --job-name=H2O
#SBATCH --output=../logs/H2O_%a.out
#SBATCH --error=../logs/H2O_%a.err
#SBATCH --array=0-161
#SBATCH --time=60:00:00
#SBATCH --ntasks=1

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step

python User_cluster.py $SLURM_ARRAY_TASK_ID
