import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply inverted dropout. If mask is provided, use it; otherwise generate one.
    """
    # dropout regularization has a mask and inverted scaling to prevent over fittign 
    # it is done because if we use this then we have a lower sum hence to compensate a multiplier used
    # multiplier is 1/1-p only when training during inference no drop out  happpens else no
    multiplier = 1/(1-p) if training else 1
    if not training:
        return x
    
    if mask is None:
        mask = (np.random.rand(*x.shape) >= p)
        
    return x*multiplier*mask
    # mask is simply a bernoulli random vaiable we dont fix neurons 