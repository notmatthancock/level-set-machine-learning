import numpy as np

def split(keys, p=[0.6, 0.2, 0.2], rs=None):
    """
    keys: list of strings
        List of keys to split into training, validation, and testing.
    
    p: list of 3 fractions.
        The probability of being placed in the training, validation or testing.

    rs: numpy.random.RandomState
        For reproducible results.
    """
    rs = np.random.RandomState() if rs is None else rs
    mn = rs.multinomial(1, pvals=p, size=len(keys))

    datasets = {}

    for j,d in enumerate(['tr', 'va', 'ts']):
        datasets[d] = [keys[i] for i in range(len(keys)) if mn[i,j]==1]
        
    return datasets
