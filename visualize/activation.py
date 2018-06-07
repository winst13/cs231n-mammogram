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

def get_activation(model, layer, image, device = torch.device('cuda'), dtype = torch.float32):
    model.eval()
    
    # Forward pass on the convolutions
    conv_output = None
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = x.to(device=device, dtype=dtype)
    print (x.shape)
    print (model(x))
    '''
    for module_pos, module in model.features._modules.items():
        x = module(x)  # Forward
        if int(module_pos) == layer:
            print (x)
            conv_output = x  # Save the convolution output on that layer
                
    activation = conv_output.data.numpy()[0]
    #resize to 1024x1024?
    activation = np.maximum(activation, 0)
    activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation))  # Normalize between 0-1
    activation = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    return activation
    '''