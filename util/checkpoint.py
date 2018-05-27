import os
import shutil
import torch

'''
Loads the model at the given path
https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
'''
def load_model(filepath, model, optimizer):
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
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
def save_model(state, accuracy, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print ("saved checkpoint to ", filename)
    with open("best_stats.txt", 'r') as best_file:
        accuracy = best_file.read()
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print ("best checkpoint! saved to ", 'model_best.pth.tar')
        with open("best_stats.txt", 'w') as best_file:
            best_file.write(accuracy)
