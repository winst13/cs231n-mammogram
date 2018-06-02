import torch
from util.util import print



def get_saliency_map(model, x):
    """
    Params:
        :model: A module, already trained. we freeze weights and get input gradients
        :x: The input image tensor (-1, 1, 1024, 1024).
    """
    x = torch.tensor(x) # If type(x) is not torch.tensor
    x.requires_grad = True

    for p in model.parameters():
        p.requires_grad = False
    
    model(x)
    model.backward()

    pass
    #return
