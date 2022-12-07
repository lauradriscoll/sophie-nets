#!/bin/bash
#
#SBATCH --job-name=gru
#
#SBATCH --time=24:00:00
#!/bin/bash
#SBATCH -c 1

ml python/2.7.13
module load py-scipystack/1.0_py27 py-tensorflow/1.9.0_py27
srun python general_model_train.py --gres
