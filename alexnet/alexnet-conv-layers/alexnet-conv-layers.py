import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    return np.zeros((image.shape[0],55,55,96))
    # only 1 channel now? or batch  batch