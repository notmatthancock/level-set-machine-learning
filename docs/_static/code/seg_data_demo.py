import h5py
from pylidc.utils import volume_viewer

# open in read mode, not write mode!
hf = h5py.File('dataset.h5', 'r')

img = hf['000/img'][...]
seg = hf['000/seg'][...]

print("Pixel spacing = %.2f." % hf['000'].attrs['delta_ij'][...])
print("Slice spacing = %.2f." % hf['000'].attrs['delta_k'][...])

# The creates an interactive viewer instance:
volume_viewer(img, mask=seg, c='r')
