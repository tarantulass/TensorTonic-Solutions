import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x) # apply vectorized approqch on vectors not list
    return 1 / (1 + np.exp(-x))