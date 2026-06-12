import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m)
    grad = np.array(grad)
    param = np.array(param)
    v = np.array(v)
    
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2 # moment m and velocity is second moment
    # bias correction
    mhat  = m/(1-beta1**t)
    vhat = v/(1-beta2**t)

    param = param - lr*(mhat)/(np.sqrt(vhat) + eps)
    # we dont return bias corrected versions in Adam
    return param, m, v # 2nd moment after 1st