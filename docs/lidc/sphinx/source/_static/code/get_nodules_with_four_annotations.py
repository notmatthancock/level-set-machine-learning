import numpy as np
import pylidc as pl

# Each element of this list will be a 4-tuple containing
# the pylidc.Annotation ids of four Annotation object which
# have been estimated to refer to the same nodule in a scan.
ann_ids = []

scans = pl.query(pl.Scan)
N = scans.count()

for i,scan in enumerate(scans):
    print("%04d / %04d" % (i+1, N))
    
    # Group annotations that are close for this scan.
    ann_groups = scan.cluster_annotations()

    for group in ann_groups:
        if len(group) == 4:
            ann_ids.append([ann.id for ann in group])

np.save('annotations_ids_group_size_4.npy', ann_ids)
