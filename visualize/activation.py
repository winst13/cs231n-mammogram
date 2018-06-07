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
    conv_output = []
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = x.to(device=device, dtype=dtype)
    
    for module_pos, module in model.features._modules.items():
        if "denseblock" in module_pos:
            for module_1 in module.modules():
                print (module_1)
                if layer in module_1:
                    x = module_1(x)
                    conv_output.append(x)
                elif "denselayer" in module_1:
                    for module_2 in module_1.modules():
                        if layer in module_2:
                            x = module_2(x)
                            conv_output.append(x)
                        else:
                            x = module_2(x)
                else:
                    x = module_1(x)
        elif layer in module_pos:
            x = module(x)
            conv_output.append(x)  # Save the convolution output on that layer
        else:
            x = module(x)
        
    for output in conv_output:
        print (output.shape)
        activation = output.cpu().data.numpy()[0]
        #resize to 1024x1024?
        activation = np.maximum(activation, 0)
        activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation))  # Normalize between 0-1
        activation = np.uint8(activation * 255)  # Scale between 0-255 to visualize
        print (activation)
        return activation
    