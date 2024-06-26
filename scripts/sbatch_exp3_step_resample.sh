#!/bin/sh

#SBATCH -J DDD-imbalanced-step-resample
#SBATCH -o /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_3.log # TO BE EDITED
#SBATCH -e /home/dhkim0317/DoubleDescent/scripts/log_sbatch_exp3_3_error.log
#SBATCH --time 3-00:00:00

#SBATCH -p A100-80GB 
#SBATCH --gres=gpu:4

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --comment ImbalDDDstepresample

numGPU=4 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"
source /home/dhkim0317/anaconda3/etc/profile.d/conda.sh

conda activate DL

echo "Call start"
chmod 755 /home/dhkim0317/DoubleDescent/scripts/exp3_step_resample.sh

/home/dhkim0317/DoubleDescent/scripts/exp3_step_resample.sh $numGPU
echo "Call end"
