from lib.ModelWrapper import ModelWrapper
# from tensorboardX import SummaryWriter/
import torch
import argparse
from torchvision import transforms, datasets
from archs.cifar10 import vgg, resnet
import numpy as np
import random
import os
from tqdm import tqdm
import wandb
from utils.data_preprocessing.imbalancing import *

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_name', type=str, default='cifar10', help='Cifar10')
parser.add_argument('--model_name', type=str, default='resnet', help='vgg16, resnet')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--train_batch_size', type=int, default=128, help='train_batch_size')
parser.add_argument('--train_epoch', type=int, default=4000, help='train_epoch')
parser.add_argument('--eval_batch_size', type=int, default=256, help='eval_batch_size')
parser.add_argument('--label_noise', type=float, default=0.15, help='label_noise')
parser.add_argument('--num_noised_class', type=int, default=10, help='The number of classes that will receive noise. (0 < num_noised_class <= # class)')
parser.add_argument('--img_noise', type=str, default=None, help='None , Partial , All')
parser.add_argument('--k', type=int, default=64, help='1 to k iteration')
parser.add_argument('--is_lmbalance', type=bool, default=False, help='True, False')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'Resample', 'Reweight', 'DRW'])
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--extra_name', default='imbalanced', type=str, help='(additional) name to indicate experiment')

args = parser.parse_args()

data_dir = "./dataset"

# setting

###### wandb ######

wandb.init(project='DDD')
cfg = {
"model" : args.model_name,
"dataset" : args.data_name,
"learning_rate": args.lr,
"label_noise": args.label_noise,
"num_noised_class": args.num_noised_class,
"img_noise": args.img_noise,
}
wandb.config.update(cfg)

wandb.run.name = f'{args.model_name}_{args.data_name}_ln{args.label_noise}_nc:{args.num_noised_class}_{args.img_noise}'
wandb.run.save()

####################

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

# dataset = datasets.CIFAR10

"""
test_data = dataset(f'{data_dir}', train=False, transform=eval_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
                                        drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=args.train_batch_size, shuffle=False, num_workers=0,
                                        drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                                        drop_last=False)
"""

mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
# std = [0.2470, 0.2435, 0.2616] if args.dataset.startswith('cifar') else [.5, .5, .5]
# 이게 맞는 std 아닌가;

noise_transform = transforms.Compose([gauss_noise_tensor,])
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),])
train_noise_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                gauss_noise_tensor,])
eval_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std),])

if args.data_name == 'cifar10':
    train_dataset = ImbalanceCIFAR10(
        root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
        rand_number=args.rand_number, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data_path,
                                    train=False, download=True, transform=eval_transform)
    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                                num_workers=0, drop_last=False)
elif args.data_name == 'cifar100':
    train_dataset = ImbalanceCIFAR100(
        root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
        rand_number=args.rand_number, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=args.data_path,
                                    train=False, download=True, transform=eval_transform)
    train_sampler = None
    if args.train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                                num_workers=0, drop_last=False)
else:
    raise NotImplementedError(f"Dataset {args.data_name} is not supported!")

for k in range(1,65):
    print(f"\nTrain K={k} Start!\n")
    if args.model_name == 'vgg16':
        model = vgg.vgg16_bn()
    elif args.model_name == 'resnet':
        model = resnet.resnet18(k=k)
    else:
        raise Exception("No such model!")
    
    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # train the model
    save_path = os.path.join('runs', args.data_name, "{}_{}_k{}".format(args.model_name, int(args.label_noise*100), k))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)

    itr_index = 1

    cls_num_list = train_dataset.get_cls_num_list()

    # train loop
    if args.train_rule == 'Reweight':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    elif args.train_rule == 'DRW':
        idx = id_epoch // 160
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    else:
        per_cls_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=per_cls_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wrapper = ModelWrapper(model, optimizer, criterion, device)

    for id_epoch in tqdm(range(args.train_epoch)):
        wrapper.train()

        for id_batch, (inputs, targets) in enumerate(train_loader):

            loss, acc, _ = wrapper.train_on_batch(inputs, targets)
            itr_index += 1

        wrapper.eval()
        test_loss, test_acc = wrapper.eval_all(test_loader)
        print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, args.train_epoch, id_batch + 1, len(train_loader)))
        print("clean: loss={}, acc={}".format(test_loss, test_acc))
        print()

        state = {
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': test_acc,
            'epoch': id_epoch,
            'itr': itr_index
        }
        torch.save(state, os.path.join(save_path, "ckpt.pkl"))
        # return to train state.
        
    result_test_loss, result_test_acc = wrapper.eval_all(test_loader)
    print(f"#### K : {k}-Evaluation ####")
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, args.train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))

    ###### wandb ######
    wandb.log({"DD_test acc" : result_test_acc}, step = k)
    wandb.log({"DD_test loss" : result_test_loss}, step = k)
