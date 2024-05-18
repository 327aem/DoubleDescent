#!/bin/sh

#SBATCH -J DDD-imbalanced
#SBATCH -o /home/donggeonlee/repo/ImbalanceDDD/scripts/sbatch_exp3_1.log # TO BE EDITED
#SBATCH -e /home/donggeonlee/repo/ImbalanceDDD/scripts/sbatch_exp_3_1_error.log
#SBATCH --time 3-00:00:00

#SBATCH -p A100-80GB 
#SBATCH --gres=gpu:4

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --comment ImbalDDD

numGPU=4 # TO BE EDITED

echo "numGPU $numGPU"

echo "Activate conda"
source /home/donggeonlee/miniconda3/bin/activate ddd # TO BE EDITED

echo "Call start"
chmod 755 /home/donggeonlee/repo/ImbalanceDDD/scripts/exp3.sh

/home/donggeonlee/repo/ImbalanceDDD/scripts/exp3.sh $numGPU
echo "Call end"
