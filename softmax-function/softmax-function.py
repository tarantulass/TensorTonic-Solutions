import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    # for faster caluclation and avoiding overflow
    softmax = []
    # m,n = x.shape[0], x.shape[1]
    if x.ndim==2: # doesnt work for 3d hence instead of>1 use ==2 
        for i in range(x.shape[0]): # calculate for each row
            x[i] = x[i] - max(x[i])
            softmax.append(np.exp(x[i])/sum(np.exp(x[i])))
        softmax = np.array(softmax)
    else:
        # return np.array([np.exp(i) for i in x])
        # exactly the above is not required!!
        x = x - max(x)
        softmax =  np.exp(x)/sum(np.exp(x))
        
    return softmax