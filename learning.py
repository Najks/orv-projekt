import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

transforms = transforms.Compose([
    transforms.toTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])