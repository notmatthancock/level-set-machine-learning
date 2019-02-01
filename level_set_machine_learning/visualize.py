import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

from matplotlib.widgets import Slider


def interactive2d(
        img, u, seg=None, scores=None,
        img_kwargs=dict(cmap=plt.cm.gray, aspect=1, interpolation='bilinear'),
        u_kwargs=dict(c='b', ls='-', lw=2),
        seg_kwargs=dict(c='r', ls='-', lw=2)):
    """ Generate an interactive 2D viewer

    Parameters
    ----------
    img: ndarray, ndim=2
        An image.

    u: ndarray, shape=(niters,)+img.shape

    seg: ndarray, shape=img.shape

    scores: ndarray, shape=(niters,)
        An array of scores for each iterate of `u`.

    img_kwargs: args
        Any keyword arguments that can be passed to `matplotlib.pyplot.imshow`.

    u_kwargs: args
        Any keyword arguments that can be passed to `matplotlib.pyplot.plot`.

    seg_kwargs: args
        Any keyword arguments that can be passed to `matplotlib.pyplot.plot`.
    """
    if img.ndim != 2:
        raise TypeError("`img` must be 2d.")

    if u.shape[1:] != img.shape:
        raise ValueError("`u` must be shape `(niters,)+img.shape`.")
    niters = u.shape[0]

    if seg is not None:
        if seg.dtype != bool:
            raise TypeError("`seg` was not bool type.")
        if img.shape != seg.shape:
            raise ValueError("Shape mismatch between `img` and `seg`.")

    fig = plt.figure(figsize=(6, 6))
    aximg = fig.add_axes([0.1, 0.15, 0.8, 0.8])  # [left,bottom,width,height]
    aximg.imshow(img, **img_kwargs)
    aximg.axis('off')

    if seg is not None:
        seg_lines = []
        for contour in find_contours(seg.astype(np.float), 0.5):
            line = aximg.plot(contour[:, 1], contour[:, 0], **seg_kwargs)[0]
            seg_lines.append(line)

    u_lines = []
    for i in range(niters):
        # Each iteration may have multiple contours which
        # are contained in `u_line`.
        u_line = []
        for contour in find_contours(u[i], 0):
            line = aximg.plot(contour[:, 1], contour[:, 0], **u_kwargs)[0]
            line.set_visible(False)
            u_line.append(line)
        u_lines.append(u_line)

    iter_init = int(niters*0.25)
    str_iter = "Iter (%%0%dd / %d)" % (len(str(niters-1)), niters-1)

    axiter = fig.add_axes([0.1, 0.04, 0.6, 0.05])
    slider_iter = Slider(axiter, '', 0, niters-1,
                         valinit=iter_init, valstep=1, valfmt=str_iter)

    def _update(i):
        for iul, u_line in enumerate(u_lines):
            for c in u_line:
                c.set_visible(iul == int(i))
        if scores is not None:

            aximg.set_title("Overlap = %.5f"%scores[int(i)])
        fig.canvas.draw_idle()
    slider_iter.on_changed(_update)

    _update(iter_init)
    plt.show()
