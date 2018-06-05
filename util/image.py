import numpy as np
import torchvision

from os.path import join
from scipy.misc import imsave

import matplotlib.pyplot as plt


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
