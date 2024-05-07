import numpy as np


def islistlike(obj):
    return (
        isinstance(obj, list) or isinstance(obj, np.ndarray) or isinstance(obj, tuple)
    )
