import os
import numpy as np
import torch
from torchvision import datasets, transforms
f = np.load("/data/mlsnrs/zjm/MedMNIST/breastmnist.npz")
for line in f:
    print(line)