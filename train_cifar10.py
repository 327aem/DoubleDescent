from lib.ModelWrapper import ModelWrapper
# from tensorboardX import SummaryWriter/
import torch
import argparse
from torchvision import transforms, datasets
from archs.cifar10 import vgg, resnet
import numpy as np
import random
import os
import wandb

import os

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_name', type=str, default='cifar10', help='Cifar10')
parser.add_argument('--model_name', type=str, default='resnet', help='resnet , CNN')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--train_batch_size', type=int, default=128, help='train_batch_size')
parser.add_argument('--train_epoch', type=int, default=4000, help='train_epoch')
parser.add_argument('--eval_batch_size', type=int, default=256, help='eval_batch_size')
parser.add_argument('--label_noise', type=float, default=0.15, help='label_noise')
parser.add_argument('--num_noised_class', type=int, default=10, help='The number of classes that will receive noise. (0 < num_noised_class <= # class)')
parser.add_argument('--k', type=int, default=64, help='1 to k iteration')

args = parser.parse_args()

data_dir = "./dataset"

# setting

###### wandb ######

wandb.init(project='DDD')
cfg = {
"model" : args.model_name,
"dataset" : args.data_name,
"learning_rate": args.lr,
"label_noise": args.label_noise
}
wandb.config.update(cfg)

wandb.run.name = f'{args.model_name}_{args.data_name}_ln{args.label_noise}'
wandb.run.save()

###### ###### ######

dataset = datasets.CIFAR10
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

eval_transform = transforms.Compose([transforms.ToTensor()])


for k in range(1,65):
    if args.model_name == 'vgg16':
        model = vgg.vgg16_bn()
    elif args.model_name == 'resnet':
        model = resnet.resnet18(k=k)
    else:
        raise Exception("No such model!")

    # load data
    train_data = dataset(f'{data_dir}', train=True, transform=train_transform,download=True)
    train_targets = np.array(train_data.targets)
    data_size = len(train_targets)

    all_class = np.unique(train_targets) # list of all unique classes
    num_class = len(all_class) # number of classes

    # noising data
    assert args.num_noised_class <= num_class, "num_noised_class must be less than or equal to the number of classes."
    assert args.num_noised_class > 0, "num_noised_class must be greater than 0."
    num_random_elements = int(data_size*args.label_noise) # number of elements that should be noised

    if args.num_noised_class == num_class:
        random_index = random.sample(range(data_size), num_random_elements)
        random_part = train_targets[random_index]
        np.random.shuffle(random_part)
        train_targets[random_index] = random_part
        train_data.targets = train_targets.tolist()

        noise_data = dataset(f'{data_dir}', train=True, transform=train_transform)
        noise_data.targets = random_part.tolist()
        noise_data.data = train_data.data[random_index]
    else: # 0 < num_noised_class < num_class
        candidate_class = random.sample(list(all_class), args.num_noised_class) # randomly selected classes that will receive noise

        candidate_random_index = []
        candidate_random_part = []
        for i in candidate_class:
            candidate_random_index.extend(np.where(train_targets == i)[0].tolist()) # index of the selected class
            candidate_random_part.extend(train_targets[np.where(train_targets == i)[0]].tolist()) # targets of the selected class
        
        assert len(candidate_random_index) == len(candidate_random_part), "The number of index and targets must be the same."
        assert len(candidate_random_index) >= num_random_elements, \
            f"When the total number of dataset is {data_size} and `label_noise` is {args.label_noise}, the number of noised elements would be {num_random_elements}, but the number of {args.num_noised_class} classes of data is {len(candidate_random_index)}. \
                You should increase the `num_noised_class` or decrease the `label_noise`."

        random_index = random.sample(candidate_random_index, num_random_elements)
        random_part = train_targets[random_index]
        random_part = np.array(random_part)
        np.random.shuffle(random_part)
        train_targets[random_index] = random_part
        train_data.targets = train_targets.tolist()

        noise_data = dataset(f'{data_dir}', train=True, transform=train_transform)
        noise_data.targets = random_part.tolist()
        noise_data.data = train_data.data[random_index]


    test_data = dataset(f'{data_dir}', train=False, transform=eval_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=0,
                                            drop_last=False)
    noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=args.train_batch_size, shuffle=False, num_workers=0,
                                            drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                                            drop_last=False)

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wrapper = ModelWrapper(model, optimizer, criterion, device)

    # train the model
    save_path = os.path.join('runs', args.data_name, "{}_{}_k{}".format(args.model_name, int(args.label_noise*100), k))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)

    itr_index = 1
    wrapper.train()

    for id_epoch in range(args.train_epoch):
        # train loop

        for id_batch, (inputs, targets) in enumerate(train_loader):

            loss, acc, _ = wrapper.train_on_batch(inputs, targets)
            print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
                format(id_epoch+1, args.train_epoch, id_batch+1, len(train_loader), loss, acc))
            # if itr_index % 20 == 0:
                # wandb.log({
                #     "train acc" : acc}, step = itr_index)
                # wandb.log({
                #     "train loss" : loss}, step = itr_index)

            itr_index += 1

        wrapper.eval()
        test_loss, test_acc = wrapper.eval_all(test_loader)
        noise_loss, noise_acc = wrapper.eval_all(noise_loader)
        print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, args.train_epoch, id_batch + 1, len(train_loader)))
        print("clean: loss={}, acc={}".format(test_loss, test_acc))
        print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
        print()
        # wandb.log({
        #             "test acc" : test_acc}, step = itr_index)
        # wandb.log({
        #             "test loss" : test_loss}, step = itr_index)
        # wandb.log({
        #             "noise acc" : noise_acc}, step = itr_index)
        # wandb.log({
        #             "noise loss" : noise_loss}, step = itr_index)

        state = {
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': test_acc,
            'epoch': id_epoch,
            'itr': itr_index
        }
        torch.save(state, os.path.join(save_path, "ckpt.pkl"))
        # return to train state.
        wrapper.train()
    result_test_loss, result_test_acc = wrapper.eval_all(test_loader)
    result_noise_loss, result_noise_acc = wrapper.eval_all(noise_loader)
    print(f"#### K : {k}-Evaluation ####")
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, args.train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
    wandb.log({"DD_test acc" : result_test_acc}, step = k)
    wandb.log({"DD_test loss" : result_test_loss}, step = k)
    wandb.log({"DD_noise acc" : result_noise_loss}, step = k)
    wandb.log({"DD_noise loss" : result_noise_acc}, step = k)

