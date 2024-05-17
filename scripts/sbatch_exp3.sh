#!/bin/sh

#SBATCH -J DDD-imbalanced
#SBATCH -o /home/dhkim0317/DoubleDescent/scripts/sbatch_exp3_1.log # TO BE EDITED
#SBATCH -e /home/dhkim0317/DoubleDescent/scripts/sbatch_exp_3_1_error.log
#SBATCH --time 1-00:00:00

#SBATCH -p titanxp 
#SBATCH --gres=gpu:2

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --comment pytorch

numGPU=2 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate deeplearning

echo "Call start"
chmod 755 /home/dhkim0317/DoubleDescent/scripts/exp3.sh
/home/dhkim0317/DoubleDescent/scripts/exp3.sh $numGPU
echo "Call end"
