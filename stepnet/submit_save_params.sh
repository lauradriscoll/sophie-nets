#!/bin/bash
#SBATCH --job-name=save_params   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory (RAM) per node
#SBATCH --time=1:00:00
#SBATCH -c 1
ml python/2.7.13
module load py-scipystack/1.0_py27 py-tensorflow/1.9.0_py27
srun python save_pre_train_model_params.py 
