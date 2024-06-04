# DoubleDescent
Deep Learning term project : Dive into Data-driven Deep Double Descent

Usage
```python
python train_cifar10.py \
  --data_name Cifar10 \
  --model_name resnet  \
  --label_noise 0.15 \
  --num_noised_class 10 \
  --start_k 1 \ # k value to start experiment
  --end_k 64 \ # k value to end experiment
  --train_batch_size 128 \
  --train_epoch 1000
```

## Environmental
- Python >= 3.9

## References
- [ZhangXiao96/Deep-Double-Descent](https://github.com/ZhangXiao96/Deep-Double-Descent/)
- [YyzHarry/imbalanced-semi-self](https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py)
