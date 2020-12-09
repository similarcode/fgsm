import os
import torch
import FCN as fc
import numpy as np
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms

os.environ['CUDA_VISIBLE_DEVICES']='1'

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
trainset = torchvision.datasets.MNIST(root = '/data/mlsnrs/data/MNIST', train = True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root = '/data/mlsnrs/data/MNIST',train = False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64, shuffle=True,num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True,num_workers=2)

Fc = fc.Fc_net()
Fc.to(device)

def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Fc.parameters(),lr=0.001,momentum=0.9)
    for epoch in range(10):
        train=True
        al_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            optimizer.zero_grad()
            pred_y = Fc(inputs)
            loss = criterion(pred_y,labels)
            loss.backward(retain_graph=True)
            inputs_grad = inputs.grad.data
            perturb, perturbed_data = fgsm(inputs, 0.3, inputs_grad)
            pred_per = Fc(perturbed_data)
            print('pred_per')
            print(pred_per.shape)
            loss_per = criterion(pred_per,labels)
            loss = loss*0.5+loss_per*0.5
            loss.backward(retain_graph=True)
            optimizer.step()
            al_loss = loss.item()
            if i%100 == 0:
                print('[{},{}] train loss is {}'.format(epoch+1, i+1, al_loss))

def test():
    total = 0
    correct_org = 0
    correct_per = 0
#    with torch.no_grad():
    for data in testloader:
        inputs,labels = data
        inputs,labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        pred_y = Fc(inputs)
        _,pred_label = torch.max(pred_y.data, 1)
        total += labels.size(0)
        correct_org += (pred_label == labels).sum().item()
        loss = F.nll_loss(pred_y, labels)
        Fc.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        perturb,perturbed_data = fgsm(inputs,0.3,data_grad)
        out_per = Fc(perturbed_data)
        _, per_label = torch.max(out_per.data, 1)
        correct_per += (per_label == labels).sum().item()
    print('test accuracy: {}'.format(100*correct_org/total))
    print('the accurac of perturbed data: {}'.format(100*correct_per/total))


def fgsm(inputs,epsilon,data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_inputs = inputs + epsilon*sign_data_grad
    perturbed_inputs = torch.clamp(perturbed_inputs,0,1)
    return sign_data_grad,perturbed_inputs
print('start train the model')
train()
print('end train')
print('start test the model')
test()
print('test end')
