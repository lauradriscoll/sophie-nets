#!/bin/bash
#
#SBATCH --job-name=32_all
#
#SBATCH --time=40:00:00
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

ml python/2.7.13
module load py-scipystack/1.0_py27 py-tensorflow/1.9.0_py27
srun python run_model_training_w_val.py --gres
