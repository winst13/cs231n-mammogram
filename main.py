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

from util.util import print
from util.dataset_class import MammogramDataset
from util.checkpoint import save_model, load_model
from model.baseline_model import BaselineModel
from model.mammogram_densenet import MammogramDenseNet
from model.helper import *

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true', help="Debug mode: ???")
parser.add_argument("--load_check", action='store_true', help="optional argument, if we want to load an existing model")
parser.add_argument("--load_best", action='store_true', help="optional argument, if we want to load an existing model")
parser.add_argument("--print_every", default = 100, type=int, help="print loss every this many iterations")
parser.add_argument("--use_cpu", action='store_true', help="Use GPU if possible, set to false to use CPU")
parser.add_argument("--batch_size", default = 10, type=int, help="Batch size")
parser.add_argument("--val_ratio", default = 0.2, type=float, help="Proportion of training data set aside as validation")
parser.add_argument("--mode", help="can be train, test, or vis")
parser.add_argument("--save_every", default = 10, type=int, help="save model at this this many epochs")
parser.add_argument("--model_file", help="mandatory argument, specify which file to load model from")
parser.add_argument("--exp_name", help="mandatory argument, specify the name of the experiment")
parser.add_argument("--model", help="mandatory argument, specify the model being used")
parser.add_argument("--lr", default=5e-3, type=float, help="learning rate")
parser.add_argument("--dropout", default=0, type=float, help="dropout rate.  higher = more dropout")
parser.add_argument("--l2reg", default=0, type=float, help="l2 regularization rate")
parser.add_argument("--augment", action='store_true', help='additional argument to add data augmentation')
args = parser.parse_args()

#Setup
debug = args.debug
load_check = args.load_check
load_best = args.load_best
print_every = args.print_every
USE_GPU = False if args.use_cpu else True
VAL_RATIO = args.val_ratio
mode = args.mode
save_every = args.save_every
exp_name = args.exp_name
model_name = args.model
learning_rate = args.lr
dropout = args.dropout
l2reg = args.l2reg
augment = args.augment

#Hyperparameters
BATCH_SIZE = args.batch_size

#Make sure parameters are valid
assert mode == 'test' or mode == 'train' or mode == 'vis' or mode == 'tiny'
assert load_check == False or load_best == False
if mode == 'vis':  assert load_check == True or load_best == True
    
# CONSTANTS
IMAGE_SIZE = 1024*1024

transform = T.Compose([
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0), interpolation=PIL.Image.BICUBIC)
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])

if augment:
    train_data = MammogramDataset("data", "train", transform=transform)
else:
    train_data = MammogramDataset("data", "train")
test_data = MammogramDataset("data", "test")

NUM_VAL = int(len(train_data)*VAL_RATIO)
NUM_TRAIN = len(train_data) - NUM_VAL
NUM_TEST = len(test_data)

loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 
                                                                                              NUM_TRAIN + NUM_VAL)))
loader_test = DataLoader(test_data, batch_size=BATCH_SIZE)
loader_tiny_train = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(101)))
loader_tiny_val = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(100, 200)))

dtype = torch.float32
if USE_GPU and torch.cuda.is_available(): #Determine whether or not to use GPU
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)


def evaluate_metrics(preds, labels):
    truepos  = ((preds == 1) * (labels == 1)).sum()
    falsepos = ((preds == 1) * (labels == 0)).sum()
    trueneg  = ((preds == 0) * (labels == 0)).sum()
    falseneg = ((preds == 0) * (labels == 1)).sum()
    return truepos, falsepos, trueneg, falseneg


'''
Take a loader, model, and optimizer.  Use the optimizer to update the model
based on the training data, which is from the loader.  Does not terminate,
saves best checkpoint and latest checkpoint
'''
def train(loader_train, loader_val, model, optimizer, epoch, loss_list = [], val_acc_list = []):
    model = model.to(device=device)
    while True:
        tot_correct = 0.0
        tot_samples = 0.0
        tot_loss = 0.0
        for t, sample in enumerate(loader_train):
            x = sample['image'].unsqueeze(1)
            y = sample['label']
            # Move the data to the proper device (GPU or CPU)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.float)

            scores = model(x).view(-1)
            print("Predicted scores are:", scores)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(scores, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()) ## Fixed. Maybe this line was the memory leak
            
            #training acc, precision, recall, etc. metrics
            num_samples = scores.size(0)
            preds = scores > 0.5
            truepos, falsepos, trueneg, falseneg = evaluate_metrics(preds, y)
            assert (truepos + falsepos + trueneg + falseneg) == num_samples
            num_correct = truepos + trueneg
            tot_correct += num_correct
            tot_samples += num_samples
            tot_loss += loss.item()

            if t % print_every == 0:
                batch_acc = float(num_correct)/num_samples
                print('Iteration %d: batch train accuracy = %06f, loss = %06f'%(t, batch_acc, float(loss.item())))
                
        val_acc = check_accuracy(loader_val, model)
        val_acc_list.append(val_acc)
        train_acc = float(tot_correct)/tot_samples
        if epoch % save_every == 0:
            save_model({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss_list' : loss_list,
                'val_acc_list': val_acc_list
                }, val_acc, exp_name)
        print ("EPOCH %d, val accuracy = %06f"%(epoch, float(val_acc)))
        print ("train accuracy = %06f, loss = %06f"%(train_acc, tot_loss))
        '''
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)
        '''
        epoch += 1
        
'''
Takes a data loader and a model, then returns the model's accuracy on
the data loader's data set
'''
def check_accuracy(loader, model, cutoff = 0.5):
    tot_correct, tot_samples = 0, 0
    tot_truepos, tot_falsepos, tot_trueneg, tot_falseneg = 0, 0, 0, 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for sample in loader:
            x = sample['image'].unsqueeze(1)
            y = sample['label']
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x).view(-1)
            tot_samples += scores.size(0)
            preds = scores > 0.5
            truepos, falsepos, trueneg, falseneg = evaluate_metrics(preds, y)
            tot_truepos += truepos
            tot_falsepos += falsepos
            tot_trueneg += trueneg
            tot_falseneg += falseneg
    print ("tp = %d, fp = %d, tn = %d, fn = %d, tot = %d"%(tot_truepos, tot_falsepos, tot_trueneg, tot_falseneg, tot_samples))
    assert (tot_truepos + tot_falsepos + tot_trueneg + tot_falseneg) == tot_samples
    tot_correct += tot_truepos + tot_trueneg
    acc = float(tot_correct)/tot_samples
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    '''
    return acc

betas = (0.9, 0.999)

if model_name == "baseline":
    model = BaselineModel(drop_rate=dropout)
elif model_name == "tinydense":
    model = get_tiny_densenet(swish = True, debug = debug, drop_rate=dropout)
elif model_name == "smalldense":
    model = get_small_densenet(swish = True, debug = debug, drop_rate=dropout)
elif model_name == "mediumdense":
    model = get_medium_densenet(swish = True, debug = debug, drop_rate=dropout)
elif model_name == "largedense":
    model = get_large_densenet(swish = True, debug = debug, drop_rate=dropout)
elif model_name == "reducedense":
    model = get_reduced_densenet(drop_rate=dropout)
elif model_name == "nopretraindense":
    model = get_nopretrain_densenet(drop_rate=dropout)
else:
    print ("bad --model parameter")

print (model)

if device == torch.device('cuda'):
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas = betas, weight_decay=l2reg)

epoch = 0
loss_list = []
if load_check:
    epoch, loss_list, val_acc_list = load_model(exp_name, model, optimizer, mode = 'checkpoint')
if load_best:
    epoch, loss_list, val_acc_list = load_model(exp_name, model, optimizer, mode = 'best')

if mode == 'train':
    train(loader_train, loader_val, model, optimizer, epoch, loss_list = loss_list, val_acc_list = val_acc_list)
elif mode == 'tiny':
    train(loader_tiny_train, loader_tiny_val, model, optimizer, epoch, loss_list = loss_list, val_acc_list = val_acc_list)
        

