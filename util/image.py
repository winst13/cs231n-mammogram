import numpy as np


def normalize_between(imgarray, bottom, top, dtype=None):
    """ Normalizes between two numbers, but returns ndarray
        with float values!
        Casting to dtype done at very end.
    """
    minimum = float(np.amin(imgarray))
    maximum = np.amax(imgarray)
    scale_factor = (top - bottom) / (maximum - minimum)
    final_array = (imgarray - minimum) * scale_factor + bottom
    final_array = np.clip(final_array, bottom, top) # In case of numerical instability
    if dtype is not None:
        final_array = final_array.astype(dtype)

    return final_array