# DoubleDescent
Deep Learning term project : Dive into Data-driven Deep Double Descent

Usage
```python
python train_cifar10.py \
  --data_name Cifar10 \
  --model_name vgg16 \ # vgg16 / resnet 
  --label_noise 0.15 \
  --num_noised_class 10 \
  --start_k 1 \ # k value to start experiment
  --train_batch_size 128 \
  --train_epoch 1000
```
실행하면 자동으로 k -> 1~64까지 for문 돌면서 test / noise loss / acc 찍어주고, wandb에 logging

## Environmental

- Python >= 3.9

## Train Dataset 
+ label noised CIFAR10

## Test Loss / Acc 
+ original test set 

## Noise Loss / Acc 
+ noised test set
