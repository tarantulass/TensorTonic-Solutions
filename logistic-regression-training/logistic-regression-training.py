import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    N = len(y)
    w = np.zeros(X.shape[1]) # good practice always 
    b = 0
    step = 0
    while step!=steps:
        pred = _sigmoid(X@w+b)
        w = w - lr*(1/N)*(X.T@(pred-y)) # remember proof yhat had +ve
        b = b- lr*(1/N)*np.sum(pred-y)
        step+=1
    return w,b
    # always start with loss function and write updates