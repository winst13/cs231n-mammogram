import torch
import numpy as np
import matplotlib.pyplot as plt

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
    return float(numerator)/float(denomenator)

def get_precision(truepos, falsepos, trueneg, falseneg):
    return float(truepos)/float(truepos + falsepos)

def get_recall(truepos, falsepos, trueneg, falseneg):
    return float(truepos)/float(truepos + falseneg)

def save_plot(loss_list, epochs):
    time = np.linspace(0, epochs, len(loss_list), endpoint=True)
    plt.plot(time, loss_list)
    plt.show()