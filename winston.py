import numpy as np
import torch
from torch import optim

from os.path import join
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image

from model.mammogram_densenet import MammogramDenseNet
from model import helper
from util.image import normalize_between
from util.checkpoint import load_model
from visualize.activation import get_activation, save_activations


if __name__ == "__main__":
    model = helper.get_simple_densenet()
    # This is a dummy optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas = (0.9, 0.999), weight_decay=0)
    load_model('dense_dropout', model, optimizer, mode='best') # rehydrate model
    del optimizer # goodbye
    print (model)
    model.cuda()

    filename = "Mass-Test_P_01617_LEFT_CC.npy"
    imagepath = join("data/test/1", filename)
    savepath = join("visualize_output", filename[:-4] + "_activation.png")
    
    arr = np.load(imagepath)
    plt.imshow(arr, cmap='gray')
    plt.show()
    
    conv_layer_list, conv_output = get_activation(model, "conv", arr)
    save_activations(conv_layer_list, conv_output)