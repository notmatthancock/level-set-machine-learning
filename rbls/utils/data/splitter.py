import numpy as np

def split(n, p=[0.6, 0.2, 0.2], rs=None):
    """
    Given indices [0, 1, ..., n-1], 
    
    split the indices into train/validation/testing based 
    on the provided fractions in `p`.
    """
    rs = np.random.RandomState() if rs is None else rs
    mn = rs.multinomial(1, pvals=p, size=n)

    return tuple([list(np.where(mn[:,i])[0]) for i in range(3)])
