import torch
import torch.nn as nn
import torch.nn.functional as F

class Fc_net(nn.Module):
    def __init__(self):
        super(Fc_net, self).__init__()
        self.fc1 = nn.Linear(1*28*28,300)
        self.fc2 = nn.Linear(300,150)
        self.fc3 = nn.Linear(150,10)
    def forward(self, x):
        x = x.view(-1, 1*28*28)
        lay1 = F.relu(self.fc1(x))
        lay2 = F.relu(self.fc2(lay1))
        outs = F.softmax(self.fc3(lay2))
        return outs
