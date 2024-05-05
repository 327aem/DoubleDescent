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


echo "resnet training start"
python train_cifar10.py \
--data_name Cifar10 \
--model_name resnet \
--train_batch_size 128 \
--start_k 1 \
--end_k 32 \
--is_imbalance True \
--imb_type exp \
--imb_factor 0.01 \
--train_rule Resample \
--rand_number 0 \
--extra_name imbalance