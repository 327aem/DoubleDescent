#!/bin/sh

#SBATCH -J DDD
#SBATCH -o /home/donggeonlee/repo/DoubleDescent/train.log # TO BE EDITED
#SBATCH -t 3-00:00:00

#SBATCH -p 3090
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

echo "Activate conda"
source /home/donggeonlee/miniconda3/bin/activate ddd # TO BE EDITED

echo "Login to Wandb"
wandb login YOUR_API_KEY # TO BE EDITED

model_list=(vgg16 resnet )
length=${#model_list[@]}

for (( i=0; i<${length}; i++ ));
do
    python train_cifar10.py \
    --data_name Cifar10 \
    --model_name ${model_list[$i]} \
    --label_noise 0.15 \
    --batch_size 128 \
    --num_noised_class 10
done
