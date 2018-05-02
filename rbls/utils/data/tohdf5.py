"""
Utilities for converting Numpy datasets to hdf5
"""

import os
import h5py
import numpy as np
import skfmm

def convert(imgs, segs, path, dx=None, verbose=True, compress=True):
    """
    Convert a dataset of images and boolean segmentations
    to hdf5 format, which is required for the level set routine.

    Format assuming `hf` is and h5py `File`:
        
        i
        |_ img
        |_ seg
        |_ dist
        |_ attrs
           |_ dx

    In other words, the 17th image example would be accessed like::

        img17 = hf["17/img"][...]

    The attributes (`attrs`) are special and accessed like::
        
        hf["17"].attrs["dx"]

    The format above is the one necessary for level set routine; however,
    additional information can be stored in the dataset if desired by 
    writing separate scripts to append the dataset. For example, the center
    of mass of the ground truth might be useful meta (attribute) information 
    to store.

    Parameters
    ----------
    imgs: ndarray, shape=(nexamples,) + img.shape
        The list of image examples for the dataset.

    segs: ndarray, shape=(nexamples,) + seg.shape
        The list of image examples for the dataset.

    path: str
        The file path and file name to store the hdf5 data file, e.g.::

            "/home/user/Data/my_dataset.h5"

    dx: ndarray, shape=(nexamples, img.ndim), default=None
        The resolutions along each axis for each image. The default (None)
        assumes the resolution is 1 along each axis direction, but this 
        might not be the case for anisotropic data (e.g., CT scan data).

    verbose: bool, default=True
        Print progress if True.

    compress: bool, default=True
        If True, `gzip` compression with default compression options (level=4)
        is used for the image and segmentations.
    """
    # Check if the file already exists and abort if so.
    if os.path.exists(path):
        raise Exception("Dataset already exists at %s." % path)

    # Ensure imgs and segs are proper.
    if imgs.dtype != np.float:
        raise TypeError("`imgs` should be float dtype.")
    if segs.dtype != np.bool:
        raise TypeError("`segs` should be bool dtype.")
    if imgs.shape != segs.shape:
        raise ValueError("`imgs` shape doesn't match `segs`")
    
    N = imgs.shape[0]
    ndim = imgs[0].ndim

    # Check dx if provided and is correct shape.
    if dx is not None and dx.shape != (N, ndim):
        raise ValueError("`dx` is wrong shape.")

    hf = h5py.File(path, 'w')

    compress = "gzip" if compress else None

    if verbose:
        q = len(str(N))
        pstr = "Creating hdf5 dataset ... %%0%dd / %d" % (q, N)

    o = np.ones(ndim, dtype=np.float)

    for i in range(N):
        if verbose: print(pstr % (i+1))

        # Create a group for the i'th example
        g = hf.create_group(str(i))

        # Store the i'th image and segmentation.
        g.create_dataset("img", data=imgs[i], compression=compress)
        g.create_dataset("seg", data=segs[i], compression=compress)

        # Compute the signed distance transform of the ground-truth
        # segmentation and store it.
        dist = skfmm.distance(2*segs[i].astype(np.float)-1,
                              dx=(o if dx is None else dx[i]))

        g.create_dataset("dist", data=dist, compression=compress)

        # Store the delta terms as an attribute.
        g.attrs['dx'] = o if dx is None else dx[i]

    hf.close()
    if verbose: print("Success: hdf5 dataset saved at %s." % path)
