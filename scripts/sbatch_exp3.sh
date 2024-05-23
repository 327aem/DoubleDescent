#!/bin/sh

#SBATCH -J DDD-imbalanced
#SBATCH -o /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_1.log # TO BE EDITED
#SBATCH -e /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_1_error.log
#SBATCH --time 3-00:00:00

#SBATCH -p A100-80GB 
#SBATCH --gres=gpu:3

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --comment ImbalDDD

numGPU=3 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"
source /home/dhkim0317/anaconda3/etc/profile.d/conda.sh

conda activate DL

echo "Call start"
chmod 755 /home/dhkim0317/DoubleDescent/scripts/exp3.sh

/home/dhkim0317/DoubleDescent/scripts/exp3.sh $numGPU
echo "Call end"
