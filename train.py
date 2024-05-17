from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from os import path
import argparse

import resnet

# Data
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

trainset = torchvision.datasets.CIFAR10(
    root='/local_datasets/cifar_10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./local_datasets/cifar_10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet.resnet20()
net = net.to(device)
summary(net,(3,32,32))
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)

# learning rate is 0.1 when iterations is smaller than 32k
# learning rate is 0.01 when iterations are between 32k and 48k
# learning rate is 0.001 when iterations are larger than 48k
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[32000,48000], gamma=0.1)


# Training
writer = SummaryWriter(path.join('log', 'train'), flush_secs=1)
def train(epoch,iter):
    print(f"Epoch : {epoch}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iter += 1
    writer.add_scalar("Loss_train", train_loss/(batch_idx+1), iter)
    writer.add_scalar("train_error",100.*(1.-correct/total),iter)
    print(f"Iteraions : {iter}")
    return iter 

def test(epoch,iter):
    global lowest_test_error
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    writer.add_scalar("Loss_test", test_loss/(batch_idx+1), iter)
    writer.add_scalar("test_error",100.*(1.-correct/total), iter)
    if lowest_test_error > 100.*(1.-correct/total):
        lowest_test_error = 100.*(1.-correct/total)
    
iter = 0
start_epoch = 0
lowest_test_error = 100
for epoch in range(start_epoch, start_epoch+200):
    iter = train(epoch,iter)
    test(epoch,iter)
    if iter > 64000: # terminate training at 64k
        break
    scheduler.step()
print(f"==========test error : {lowest_test_error}===========")
writer.flush()
writer.close()
