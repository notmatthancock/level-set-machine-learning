import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import skfmm

from matplotlib.widgets import Slider


def viewer(img, u=None, seg=None, axis=-1, aspect=None, **line_kwargs):
    """
    Interactive volume viewer utility

    Parameters
    ----------
    img: ndarray, ndim=3
        An image volume.

    mask: ndarray, ndim=3, dtype=bool
        A boolean mask volume.

    axis: int, default=2
        Image slices of the volume are viewed by holding this 
        axis to integer values.

    aspect: float or :class:`pylidc.Scan` object, default=None
        If float, then the argument is passed to `pyplot.imshow`.
        If the Scan object associatd with the image volume is passed,
        then the image aspect ratio is computed automatically. This is useful
        when the axis != 2, where the image slices are most likely 
        anisotropic.

    line_kwargs: args
        Any keyword arguments that can be passed to `matplotlib.pyplot.plot`.

    Example
    -------
    An example::

        import pylidc as pl
        from pylidc.utils import volume_viewer

        ann = pl.query(pl.Annotation).first()
        vol = ann.scan.to_volume()

        padding = 70.0

        mask = ann.boolean_mask(pad=padding)
        bbox = ann.bbox(pad=padding)

        volume_viewer(vol[bbox], mask, axis=0, aspect=ann.scan,
                      ls='-', lw=2, c='r')

    """
    if vol.ndim !=3:
        raise TypeError("`vol` must be 3d.")
    if axis not in (0,1,2):
        raise ValueError("`axis` must be 0,1, or 2.")
    if mask is not None:
        if mask.dtype != bool:
            raise TypeError("mask was not bool type.")
        if vol.shape != mask.shape:
            raise ValueError("Shape mismatch between image volume and mask.")
    if aspect is not None and isinstance(aspect, Scan):
        scan = aspect
        dij = scan.pixel_spacing
        dk  = scan.slice_spacing
        d = np.r_[dij, dij, dk]
        inds = [i for i in range(3) if i != axis]
        aspect = d[inds[0]] / d[inds[1]]
    else:
        aspect = 1.0 if aspect is None else float(aspect)

    nslices = vol.shape[axis]
    k = int(0.5*nslices)

    # Selects the jth index along the correct axis.
    slc = lambda j: tuple(np.roll([j, slice(None), slice(None)], axis))

    fig = plt.figure(figsize=(6,6))
    aximg = fig.add_axes([0.1,0.2,0.8,0.8])
    img = aximg.imshow(vol[slc(k)], vmin=vol.min(), aspect=aspect,
                       vmax=vol.max(), cmap=plt.cm.gray)
    aximg.axis('off')

    if mask is not None:
        contours = []
        for i in range(nslices):
            contour = []
            for c in find_contours(mask[slc(i)].astype(np.float), 0.5):
                line = aximg.plot(c[:,1], c[:,0], **line_kwargs)[0]
                line.set_visible(0)
                contour.append(line)
            contours.append(contour)

    axslice = fig.add_axes([0.1, 0.1, 0.8, 0.05])
    axslice.set_facecolor('w')
    sslice = Slider(axslice, 'Slice', 0, nslices-1,
                    valinit=k, valstep=1)
    
    def update(i):
        img.set_data(vol[slc( int(i) )])
        sslice.label.set_text('%d'%i)
        if mask is not None:
            for ic,contour in enumerate(contours):
                for c in contours[ic]:
                    c.set_visible(ic == int(i))
        fig.canvas.draw_idle()
    sslice.on_changed(update)

    update(k)
    plt.show()
