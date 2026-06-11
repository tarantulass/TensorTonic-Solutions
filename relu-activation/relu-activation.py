import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.array(x)
    return np.maximum(0,x) # important numpy doesnt have any relu inbulit like torch has