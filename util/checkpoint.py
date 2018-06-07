import os
import shutil
import torch
from util.util import print

EXPERIMENT_DIR = "experiments"
STATS_FILE = "best_stats.txt"
BEST_FILE = "model_best.pth.tar"

'''
Loads the model at the given path
https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
'''
def load_model(exp_name, model, optimizer, mode = 'checkpoint', lr = None):
    if mode == 'checkpoint':
        filename = 'checkpoint.pth.tar'
    elif mode == 'best':
        filename = 'model_best.pth.tar'
        
    filepath = os.path.join(EXPERIMENT_DIR, exp_name, filename)
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if lr is not None:
            checkpoint['optimizer']['lr'] = lr
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_list = checkpoint['loss_list']
        val_acc_list = checkpoint['val_acc_list']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filepath, checkpoint['epoch']))
        return epoch, loss_list, val_acc_list
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
        return None

'''
Turns a model and its parameters into a file
https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
Uses accuracy to figure out best checkpoint
'''
def save_model(state, acc, exp_name, filename='checkpoint.pth.tar'):
    file_path = os.path.join(EXPERIMENT_DIR, exp_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    torch.save(state, os.path.join(file_path, filename))
    print ("saved checkpoint to ", file_path)
    best_stats_file = os.path.join(file_path, STATS_FILE)
    if os.path.isfile(best_stats_file):
        with open(best_stats_file, 'r') as best_file:
            best_acc = best_file.read()
        if float(best_acc) < acc:
            shutil.copyfile(os.path.join(file_path, filename), os.path.join(file_path, BEST_FILE))
            print ("best checkpoint! saved to ", BEST_FILE)
            with open(best_stats_file, 'w') as best_file:
                best_file.write("%f"%(acc))
    else:
        shutil.copyfile(os.path.join(file_path, filename), os.path.join(file_path, BEST_FILE))
        print ("best checkpoint! saved to ", BEST_FILE)
        with open(best_stats_file, 'w') as best_file:
            best_file.write("%f"%(acc))
