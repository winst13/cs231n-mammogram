import numpy as np
import torch

# Project imports
from model.mammogram_densenet import MammogramDenseNet
from model.helper import *
from util.image import normalize_between, saliency2imgarray


if __name__ == "__main__":
    # testarr = np.random.rand(2,1024,1024)
    # normed = normalize_between(testarr, 10, 11, batch=True)
    # print(normed)

    model = get_nopretrain_densenet(debug=True)
    test = model(torch.rand(1,1,1024,1024))
    print(model)
