import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from util.dataset_class import MammogramDataset

# PARAMETERS
print_every = 1 # Constant to control how frequently we print train loss
USE_GPU = True #Use GPU if available
BATCH_SIZE = 10

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

train_data = MammogramDataset("data", "train")#Can add transform = transform as an argument
val_data = MammogramDataset("data", "train")
test_data = MammogramDataset("data", "test")

VAL_RATIO = 0.2
NUM_VAL = int(len(train_data)*VAL_RATIO)
NUM_TRAIN = len(train_data) - NUM_VAL
NUM_TEST = len(test_data)

loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(val_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 
                                                                                              NUM_TRAIN + NUM_VAL)))
loader_test = DataLoader(test_data, batch_size=BATCH_SIZE)

dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available(): #Determine whether or not to use GPU
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

for t, sample in enumerate(loader_train):
    # Move the data to the proper device (GPU or CPU)
    x = sample['image']
    y = sample['label']
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=torch.long)

    if t % print_every == 0:
        print('Iteration %d' % (t))
        print('y = ', y.shape)
        print('x shape = ', x.shape)
        print()

