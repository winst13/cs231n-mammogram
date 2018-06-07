import numpy as np
import torch
from torch import optim

from os.path import join
import matplotlib.pyplot as plt
from PIL import Image

from model.mammogram_densenet import MammogramDenseNet
from model import helper
from util.image import normalize_between
from util.checkpoint import load_model


def get_activation(model, layer, image):
    activation = image
    return activation