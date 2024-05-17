#!/bin/sh

#SBATCH -J DDD-imbalanced
#SBATCH -o /home/dhkim0317/DoubleDescent/sbatch_exp3_1.log # TO BE EDITED
#SBATCH -t 24:00:00

#SBATCH -p A100-80GB
#SBATCH --gres=gpu:4

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16

numGPU=4 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"
source /home/dhkim0317/miniconda3/bin/activate deeplearning # TO BE EDITED

echo "Call start"
/home/dhkim0317/DoubleDescent/scripts/exp3.sh $numGPU
echo "Call end"