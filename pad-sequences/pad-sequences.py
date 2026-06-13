import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # pad value fills with that no. 
    N = len(seqs)
    # seqs = np.array(seqs)
    # inhomogeneous shape not good to convert to np array
    if max_len is None:
        max_len = 0
        for i in range(N):
            max_len = max(len(seqs[i]), max_len)
            
    seqs = [np.pad(seq[:max_len], pad_width = (0,max(0,max_len-len(seq))), constant_values=pad_value)
            for seq in seqs]
        
    return seqs