import numpy as np

def split(keys, p=[0.6, 0.2, 0.2], subset_size=None, rs=None):
    """
    Split a list `keys` randomly into training, validation, and testing sets.

    Parameters
    ----------
    keys: list of strings
        List of keys to split into training, validation, and testing.
    
    p: list of 3 fractions.
        The probability of being placed in the training, validation or testing.

    subset_size: int, default=None
        If supplied, then should be less than or equal to `len(keys)`. If 
        given, then `keys` is first subsampled by `subset_size` before 
        splitting.

    rs: numpy.random.RandomState
        For reproducible results.

    Returns
    -------
    split_keys: dict
        split_keys['tr'] = [training_key1, ...]
        split_keys['va'] = [validation_key1, ...]
        split_keys['ts'] = [testing_key1, ...]
    """
    if subset_size is not None and subset_size > len(keys):
        raise ValueError("`subset_size` must be <= `len(keys)`.")
    if subset_size is None:
        subset_size = len(keys)

    rs = np.random.RandomState() if rs is None else rs

    subkeys = rs.choice(keys, replace=False, size=subset_size)
    mn = rs.multinomial(1, pvals=p, size=len(subkeys))

    datasets = {}

    for j,d in enumerate(['tr', 'va', 'ts']):
        datasets[d] = [subkeys[i] for i in range(len(subkeys)) if mn[i,j]==1]
        
    return datasets
