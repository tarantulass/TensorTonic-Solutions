import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    pe = np.zeros((seq_len,d_model))
    for pos in range(seq_len):
        for i in range(d_model//2): # last ignored!!
            arg = pos/(base**(2*i/d_model))
            pe[pos][2*i] = np.sin(arg)
            pe[pos][2*i+1] = np.cos(arg)
        if d_model%2:
            i = d_model//2
            arg = pos/(base**(2*i/d_model))
            pe[pos][2*i] = np.sin(arg)
            # pe[pos][2*i+1] = np.cos(arg)        
            # very very important that positonal embedding depends on the od and even of d_model 
            # clealr even means both needed odd mean son extra

    return pe