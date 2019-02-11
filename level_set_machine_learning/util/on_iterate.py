""" This module provides a few simple `on_iterate` functions that can be
used in the LevelSetMachineLearning.segment member function
"""


def collect_scores(seg, score_list, score_func):
    """ Collects the scores from the iterations. Scores are appended to
    :code:`score_list` and so an empty list should be provided. Usage::

        scores = []
        score_collector = collect_scores(seg, scores, model.scorer)
        model.segment(img=img, on_iterate=[score_collector, ...])
    """

    def on_iterate(i, u):
        score_list.append(score_func(u, seg))

    return on_iterate


def plot_contours(line_kwargs=None):
    """ 2D only: plot the contours from the level set onto the provided
    matplotlib axis on each iteration. :code:`line_kwargs` is a dictionary
    of keyword arguments that, if provided, is supplied to the `plot` function
    """

    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    lines_for_iter = []
    kwargs = line_kwargs or {'color': 'red'}

    def on_iterate(i, u):
        contours = find_contours(u, 0)

        for line in lines_for_iter:
            line.remove()
        lines_for_iter.clear()

        for contour in contours:
            line = plt.plot(contour[:, 1], contour[:, 0], **kwargs)[0]
            lines_for_iter.append(line)
            plt.pause(0.1)

    return on_iterate
