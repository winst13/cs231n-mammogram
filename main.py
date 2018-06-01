import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torch.nn import functional as F

import numpy as np
import argparse

from util.dataset_class import MammogramDataset
from util.checkpoint import save_model, load_model

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true', help="Debug mode: ???")
parser.add_argument("--checkpoint", help="optional path argument, if we want to load an existing model")
parser.add_argument("--print_every", default = 100, type=int, help="print loss every this many iterations")
parser.add_argument("--use_cpu", action='store_true', help="Use GPU if possible, set to false to use CPU")
parser.add_argument("--batch_size", default = 10, type=int, help="Batch size")
parser.add_argument("--val_ratio", default = 0.2, type=float, help="Proportion of training data set aside as validation")
parser.add_argument("--mode", help="can be train, test, or vis")
parser.add_argument("--save_every", default = 10, type=int, help="save model at this this many epochs")
parser.add_argument("--model_file", help="mandatory argument, specify which file to load model from")
args = parser.parse_args()

#Setup
debug = args.debug
checkpoint = args.checkpoint
print_every = args.print_every
USE_GPU = False if args.use_cpu else True
VAL_RATIO = args.val_ratio
mode = args.mode
save_every = args.save_every

#Hyperparameters
BATCH_SIZE = args.batch_size

#Make sure parameters are valid
assert mode == 'test' or mode == 'train' or mode == 'vis'
if mode == 'vis': assert checkpoint is not None
    
# CONSTANTS
IMAGE_SIZE = 1024*1024

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

'''
Take a loader, model, and optimizer.  Use the optimizer to update the model
based on the training data, which is from the loader.  Does not terminate,
saves best checkpoint and latest checkpoint
'''
def train(loader_train, loader_val, model, optimizer):
    epoch = 1
    model = model.to(device=device)
    while True:
        for t, sample in enumerate(loader_train):
            x = sample['image'].unsqueeze(1)
            y = sample['label']
            # Move the data to the proper device (GPU or CPU)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(scores, y)
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                acc = check_accuracy(loader_train, model)
                print('Iteration ', t, ": train accuracy = ", acc)
                
        acc = check_accuracy(loader_val, model)
        if epoch % save_every == 0:
            save_model() #NEEDS WORK
        print ("EPOCH ", epoch, ", val accuracy = ", acc)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)
        epoch += 1
        
'''
Takes a data loader and a model, then returns the model's accuracy on
the data loader's data set
'''
def check_accuracy(loader, model):
    num_correct, num_samples = 0, 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for sample in loader:
            x = sample['image'].unsqueeze(1)
            y = sample['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            #print (scores, y)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

learning_rate = 1e-1
betas = (0.9, 0.999)

model = nn.Sequential(
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(512*512, 2)
)

def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0)
    if type(model) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = betas, weight_decay=1e-3)
train(loader_train, loader_val, model, optimizer)
        

