import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from train import train_val, confusion_matrix_plot
from utils import imshow, get_optimizer
from test import test
from torchsummary import summary
import numpy as np
import torchprof
from ptflops import get_model_complexity_info

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Hyper-parameters
num_epochs = 20
batch_size = 128
LR =0.01 # 0.0005
valid_size = 0.2
# 0.0001

type_model = 'cnn'
optimizer_type = 'sgd'

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_train = len(trainset)
indices = list(range(num_train))

#Randomize indices
np.random.shuffle(indices)

split = int(np.floor(num_train*valid_size))
train_index, test_index = indices[split:], indices[:split]

# Making samplers for training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(test_index)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
experiment_dir = "{}-{}-{}-{}".format(current_time, 'cifar10',
                                       type_model, optimizer_type)
    
experiment_dir = os.path.join("experiments", experiment_dir)
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)
    os.mkdir(checkpoint_dir)

writer = SummaryWriter(log_dir=experiment_dir)


if __name__ == "__main__":

    # net = ResNet18()
    # net = VGG('VGG19')
    net = CNN()
    # net = CNN_3()
    # net = CNN_4()
    # net = My_CNN()
    print(net)
    summary(net,(3,32,32))
    with torchprof.Profile(net, use_cuda = True) as prof:
        net(torch.rand([1, 3, 32, 32]).cuda())
    
    print(prof.display(show_events=False))

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    net = CNN()
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = get_optimizer(optimizer_type, net, LR)
    
    
    for epoch in range(num_epochs):
        # training and validate
        train_val(net, epoch, device, trainloader,validloader, writer, criterion, optimizer, 
                 batch_size, classes, checkpoint_dir)
        # confusion_matrix_plot
        confusion_matrix_plot(net,device,validloader, writer, epoch, classes)
    
    
    
    
    # testing
    test(net, criterion, testloader, device, writer, classes )
    

