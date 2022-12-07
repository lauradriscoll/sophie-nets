#!/bin/bash
#
#SBATCH --job-name=single_tasks
#
#SBATCH --time=10:00:00
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

nvidia-smi
ml python/2.7.13
module load py-scipy/1.1.0_py27 viz py-matplotlib/2.2.2_py27 py-numpy/1.14.3_py27 py-tensorflow/1.12.0_py27
ml py-scikit-learn/0.19.1_py27
srun python run_model_training_task_order.py --gres