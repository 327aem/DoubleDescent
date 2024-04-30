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

for model_name in vgg16 resnet;
do
    echo "${model_name} training start"
    python train_cifar10.py \
    --data_name Cifar10 \
    --model_name ${model_name} \
    --label_noise 0.15 \
    --train_batch_size 128 \
    --num_noised_class 10 \
    --start_k 1 \
    --train_epoch 1000
done
