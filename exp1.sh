#!/bin/bash

masterK=64
numGPU=$2
numNoisedClass=$1
chmod 755 /home/donggeonlee/repo/DoubleDescent/train_cifar10.py

cd /home/donggeonlee/repo/DoubleDescent
echo Current Path :
pwd

for ((k=$masterK;k>=1;k--))
do
    touch num_noised_class-$numNoisedClass-k_$k.log
    cat /dev/null > num_noised_class-$numNoisedClass-k_$k.log

    echo GPU: $(($(($k + $numNoisedClass)) % $numGPU)) - numNoisedClass: $numNoisedClass - K: $k - Start Code

    CUDA_VISIBLE_DEVICES=$(($(($k + $numNoisedClass)) % $numGPU)) python -u train_cifar10.py \
        --data_name Cifar10 \
        --model_name resnet \
        --label_noise 0.15 \
        --start_k $k \
        --end_k $k \
        --num_noised_class $numNoisedClass > num_noised_class-$numNoisedClass-k_$k.log 2>&1 &
    # echo End Code


done

tail -f num_noised_class-$numNoisedClass-k_$masterK.log                                                                        