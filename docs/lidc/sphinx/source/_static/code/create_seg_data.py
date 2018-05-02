import h5py   as h5
import numpy  as np
import pylidc as pl
from pylidc.utils import consensus

ids = np.load('./annotations_ids_group_size_4.npy')

hf = h5.File('dataset.h5', 'w')

# We will construct the volumes so that the physical dimensions of each 
# volume are at least 65 millimeters. This is because a priori, we don't 
# know the size nodule in the image volume (it would be bad to use the 
# ground truth to determine the bbox size). We do however know that the 
# largest nodule is no bigger than about 65 millimeters.
bbox_extent = 65.0


for i in range(ids.shape[0]):
    # Print progress.
    print( "[%s] %03d / %03d" % (s, i+1, N) )

    nod_grp = hf.create_group('%03d' % i)

    # Gather the 4 pylidc annotation objects for this nodule.
    anns = [pl.query(pl.Annotation).get(aid) for aid in nod]

    nod_grp.attrs['aids'] = [aid for aid in nod]

    # Store the resolutions.
    dij = anns[0].scan.pixel_spacing
    dk  = anns[0].scan.slice_spacing
    nod_grp.attrs['delta_ij'] = dij
    nod_grp.attrs['delta_k']  = dk

    # Create and store the boolean mask.
    mask, bbox = consensus(anns, pad=bbox_extent, ret_masks=False)
    nod_grp.create_dataset('seg', data=mask, compression='gzip')

    # Store the bounding box indices.
    nod_grp.attrs['bbox'] = [[b.start, b.stop-1] for b in bbox]

    # Compute and store the center of mass of the nodule.
    inds = np.indices(mask.shape, dtype=np.float)
    vol = mask.sum()*1.0
    com = np.array([(ind*mask.astype(np.float)).sum()/vol
                        for ind in inds])
    nod_grp.attrs['ground-truth-center-of-mass'] = com

    # Use the first annotation (wlog) to get scan image info
    img = anns[0].scan.to_volume(verbose=False)[bbox]

    # Normalize the image volume by its mean and std dev.
    # (it might be preferable to do this at run-time instead)
    #img = (img-img.mean()) / img.std()

    nod_grp.create_dataset('img', data=img, compression='gzip')

# Close the file.
hf.close()
