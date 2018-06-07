import torch
import numpy as np
from matplotlib import pyplot as plt

from util.util import print


def get_gradient(model, x):
    """
    Params:
        :model: A module, already trained. we freeze weights and get input gradients
        :x: The input image tensor (-1, 1, 1024, 1024).
    Return:
        :gradient: torch.Tensor (-1, 1, 1024, 1024) saliency map.
    """
    x = torch.tensor(x)
    x.requires_grad = True # gradient wrt image

    # Freeze params, we're not updating weights
    for p in model.parameters():
        p.requires_grad = False
    
    try:
        scores = model(x)
    except RuntimeError:
        model = model.cuda()
        x = x.cuda()
        x.requires_grad = True
        scores = model(x)
    scores.backward()

    gradient = x.grad
    print("We got dL/dx of shape", gradient.size())

    gradient = gradient.abs_().mean(1) # 1 = channel dim, (-1, 1024, 1024)
    print("After processing (absval + mean):", gradient.size())
    return gradient


def save_saliency_and_image(tensor, image, savepath):
    """ Take in a saliency map tensor, and output as img array. Save if option provided.
    Batch size should not exist, ideally.
    Params:
        :image: The original (1024, 1024) image
        :tensor: (1024, 1024) saliency map
    Return:
        :img: [(-1,] (1024, 1024, 1) img array, or a list of such
    """
    assert savepath.endswith(".png")
    assert savepath.startswith("visualize_output/")

    # https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
    plt.imshow(image, cmap='gray')
    plt.imshow(tensor, cmap='hot', alpha=0.7)
    plt.savefig(savepath)


def create_saliency_overlay(model, imagepath, savepath):
    """
    Params:
        :model: pytorch model for this scope, e.g. MammogramDenseNet
        :image: (1024, 1024) numpy array
    Return: None.
    """
    assert imagepath.endswith('.npy')
    image = np.load(imagepath)

    x = image.reshape(1, 1, 1024, 1024)
    raw_gradient = get_gradient(model, x)
    saliency_tensor = raw_gradient.numpy().reshape(1024, 1024)

    save_saliency_and_image(saliency_tensor, image, savepath)


