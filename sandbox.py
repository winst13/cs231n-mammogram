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
    model = helper.get_reduced_densenet()
    # This is a dummy optimizer
    optimizer = optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas = (0.9, 0.999), weight_decay=0)
    load_model('dense_060518', model, optimizer, mode='best') # rehydrate model
    del optimizer # goodbye

    filename = "Mass-Test_P_01617_LEFT_CC.npy"
    imagepath = join("visualize_input", filename)
    savepath = join("visualize_output", filename[:-4] + "_saliency.png")
    create_saliency_overlay(model, imagepath, savepath)


    """
    model = get_nopretrain_densenet(debug=True)
    test = model(torch.rand(1,1,1024,1024))
    print(model)
    """
