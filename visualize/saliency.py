import torch
from util.util import print



def get_saliency_map(model, x):
    """
    Params:
        :model: A module, already trained. we freeze weights and get input gradients
        :x: The input image tensor (-1, 1, 1024, 1024).
    Return:
        :gradient: torch.Tensor (-1, 1024, 1024) saliency map.
    """
    x = torch.tensor(x)
    x.requires_grad = True # gradient wrt image

    # Freeze params, we're not updating weights
    for p in model.parameters():
        p.requires_grad = False
    
    scores = model(x)
    scores.backward()

    gradient = x.grad
    print("We got dL/dx of shape", gradient.size())

    gradient = gradient.abs_().mean(1) # 1 = channel dim, (-1, 1024, 1024)

    return gradient
