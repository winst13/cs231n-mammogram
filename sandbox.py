import torch


# Project imports
from model.mammogram_densenet import MammogramDenseNet



if __name__ == "__main__":
    model = MammogramDenseNet(debug=True)
    model(torch.rand(1,1,1024,1024))



