#!/bin/sh

#SBATCH -J DDD-7-80
#SBATCH -o /home/donggeonlee/repo/DoubleDescent/sbatch_exp1_7.log # TO BE EDITED
#SBATCH -t 3-00:00:00

#SBATCH -p A100-80GB
#SBATCH --gres=gpu:2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16

numNoisedClass=7 # TO BE EDITED
numGPU=2 # TO BE EDITED

echo "num_noised class $numNoisedClass"

echo "Activate conda"
source /home/donggeonlee/miniconda3/bin/activate ddd # TO BE EDITED

echo "Call start"
/home/donggeonlee/repo/DoubleDescent/exp1.sh $numNoisedClass $numGPU
echo "Call end"