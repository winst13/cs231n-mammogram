import os
import shutil
import torch

EXPERIMENT_DIR = "experiments"
STATS_FILE = "best_stats.txt"

'''
Loads the model at the given path
https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
'''
def load_model(exp_name, model, optimizer, mode = 'checkpoint'):
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
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filepath, checkpoint['epoch']))
        return checkpoint, epoch, model, optimizer
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
    torch.save(state, os.path.join(filepath, filename))
    print ("saved checkpoint to ", file_path)
    best_stats_file = os.path.join(file_path, STATS_FILE)
    if os.path.isfile(best_stats_file):
        with open(best_stats_file, 'r') as best_file:
            best_acc = best_file.read()
        if best_acc < acc:
            shutil.copyfile(filename, 'model_best.pth.tar')
            print ("best checkpoint! saved to ", 'model_best.pth.tar')
            with open(best_stats_file, 'w') as best_file:
                best_file.write(acc)
    else:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print ("best checkpoint! saved to ", 'model_best.pth.tar')
        with open(best_stats_file, 'w') as best_file:
            best_file.write(acc)
