import numpy as np
import torchvision

from os.path import join
from scipy.misc import imsave


def normalize_between(imgarray, bottom, top, batch=False, dtype=None):
    """ Normalizes between two numbers, but returns ndarray
        with float values!
        Casting to dtype done at very end.
        :bottom, top: scalar floats to cap the array.
    """
    if batch:
        target_axes = tuple(range(1, len(imgarray.shape)))
    else:
        target_axes = None

    minimum = np.amin(imgarray, axis=target_axes, keepdims=True).astype(np.float32)
    maximum = np.amax(imgarray, axis=target_axes, keepdims=True)
    scale_factor = (top - bottom) / (maximum - minimum) # (-1, 1, 1)
    final_array = (imgarray - minimum) * scale_factor + bottom
    final_array = np.clip(final_array, bottom, top) # In case of numerical instability
    if dtype is not None:
        final_array = final_array.astype(dtype)

    return final_array


def saliency2imgarray(tensor, savedir=None):
    """ Take in a saliency map tensor, and output as img array. Save if option provided.
    Batch size should not exist, ideally.
    Params:
        :tensor: [(-1,] (1024, 1024) saliency map
    Return:
        :img: [(-1,] (1024, 1024, 1) img array
    """
    imgarray = tensor.view(-1, 1024, 1024, 1).numpy()
    normed = normalize_between(imgarray, 0, 256, batch=True, dtype=np.uint8) # 0-255 uint8's

    if savepath is not None:
        assert savedir.startswith("visualize_output/")
        for i in range(imgarray.shape[0]):
            filename = "saliency_%d.jpg" % i
            imsave(join(savedir, filename), img[i])

    return img



