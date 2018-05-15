import numpy as np


def normalize_between(imgarray, bottom, top, dtype=None):
    """ Normalizes between two numbers, but returns ndarray
        with float values!
    """
    minimum = float(np.amin(imgarray))
    maximum = np.amax(imgarray)
    gamut = maximum - minimum
    final_array = (imgarray - minimum) / gamut
    if dtype is not None:
        final_array = final_array.astype(dtype)

    return final_array