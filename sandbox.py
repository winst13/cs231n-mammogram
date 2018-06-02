import torch

# Project imports
from model.mammogram_densenet import MammogramDenseNet
from model.helper import *



if __name__ == "__main__":
    model = get_reduced_densenet(debug=True)
    test = model(torch.rand(1,1,1024,1024))

    print(model)