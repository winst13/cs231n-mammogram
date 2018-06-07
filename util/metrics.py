import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

asset_path = "assets/plots"

def evaluate_metrics(preds, labels):
    truepos  = ((preds == 1) * (labels == 1)).sum()
    falsepos = ((preds == 1) * (labels == 0)).sum()
    trueneg  = ((preds == 0) * (labels == 0)).sum()
    falseneg = ((preds == 0) * (labels == 1)).sum()
    return truepos, falsepos, trueneg, falseneg

def get_f_beta(truepos, falsepos, trueneg, falseneg, beta = 1):
    beta2 = beta**2
    numerator = (1+beta2)*truepos
    denomenator = (1+beta2)*truepos + beta2*falseneg + falsepos
    if denomenator == 0:
        return 0
    return float(numerator)/float(denomenator)

def get_precision(truepos, falsepos, trueneg, falseneg):
    denomenator = truepos + falsepos
    if denomenator == 0:
        return 0
    return float(truepos)/float(denomenator)

def get_recall(truepos, falsepos, trueneg, falseneg):
    denomenator = truepos + falseneg
    if denomenator == 0:
        return 0
    return float(truepos)/float(denomenator)

def save_plot(loss_list, epochs, exp_name, mode):
    time = np.linspace(0, epochs, len(loss_list), endpoint=True)
    plt.plot(time, loss_list)
    save_path = join(asset_path, exp_name, mode)
    plt.savefig(save_path, bbox_inches='tight')