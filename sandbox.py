import numpy as np
import torch
from torch import optim

from os.path import join

# Project imports
from model.mammogram_densenet import MammogramDenseNet
from model import helper
from visualize.saliency import create_saliency_overlay
from util.image import normalize_between
from util.checkpoint import load_model


if __name__ == "__main__":

    #model = helper.get_tiny_densenet(debug=True)

    #model = helper.get_simple_densenet(debug=True)

    model = helper.get_simple_densenet()
    # This is a dummy optimizer
    optimizer = optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas = (0.9, 0.999), weight_decay=0)
    load_model('simple_060718', model, optimizer, mode='best') # rehydrate model
    del optimizer # goodbye

    filename_list = [
        "Mass-Test_P_01617_LEFT_CC.npy", # This is label 1
        "Calc-Test_P_00562_RIGHT_MLO.npy", # label 0
        "Mass-Test_P_00037_RIGHT_CC.npy", #  label 1
        "Mass-Test_P_00124_RIGHT_CC.npy"  #  label 0
    ]

    for filename in filename_list:
        imagepath = join("visualize_input", filename)
        savepath = join("visualize_output", "saliency", filename[:-4], "ONLYsaliency.png")
        create_saliency_overlay(model, imagepath, savepath, only_saliency=True)


    """
    model = get_nopretrain_densenet(debug=True)
    test = model(torch.rand(1,1,1024,1024))
    print(model)
    """
