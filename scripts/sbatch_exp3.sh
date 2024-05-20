#!/bin/sh

#SBATCH -J DDD-imbalanced
#SBATCH -o /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_1.log # TO BE EDITED
#SBATCH -e /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_1_error.log
#SBATCH --time 1-00:00:00

#SBATCH -p titanxp 
#SBATCH --gres=gpu:2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --comment ImbalDDD

numGPU=4 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate deeplearning

echo "Call start"
chmod 755 /home/dhkim0317/DoubleDescent/scripts/exp3.sh

/home/dhkim0317/DoubleDescent/scripts/exp3.sh $numGPU
echo "Call end"
