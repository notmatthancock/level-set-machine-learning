import logging

import numpy

from .datasets_handler import (
    DatasetsHandler,
    TESTING_DATASET_KEY, TRAINING_DATASET_KEY, VALIDATION_DATASET_KEY)
from .exception import ModelAlreadyFit
from .temporary_data_handler import (
    LEVEL_SET_KEY, MASK_KEY, SIGNED_DIST_KEY, TemporaryDataHandler,)


logger = logging.getLogger(__name__)


def setup_logging():
    """ Sets up logging formatting, etc
    """
    logger_filename = "fit-log.txt"

    line_fmt = ("[%(asctime)s] [%(name)s:%(lineno)d] "
                "%(levelname)-8s %(message)s")

    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        filename=logger_filename, format=line_fmt,
        datefmt=date_fmt, level=logging.DEBUG)


class FitJobHandler:
    """ Manages attributes and model fitting data/procedures
    """
    def __init__(self, model, data_filename, imgs, segs, dx,
                 normalize_imgs_on_convert, datasets_split, seeds,
                 random_state, step, temp_data_dir, subset_size,
                 regression_model_class, regression_model_kwargs,
                 validation_history_len, validation_history_tol, max_iters):
        """ See :class:`level_set_machine_learning.LevelSetMachineLearning`
        for complete descriptions of parameters
        """
        if model._is_fitted:
            msg = "This model has already been fit"
            raise ModelAlreadyFit(msg)

        # Validate seeds argument
        if not callable(seeds):
            if not all([isinstance(seed, int) for seed in seeds]):
                msg = "`seeds` should list of ints or callable"
                raise TypeError(msg)

        # Store the seeds list or function
        self.seeds = seeds

        # Set up logging formatting, file location, etc.
        setup_logging()

        # The LevelSetMachineLearning instance
        self.model = model

        # Initialize the iteration number
        self.iteration = 0
        self.max_iters = max_iters

        # Store the model class and keyword arguments
        self.regression_model_class = regression_model_class
        self.regression_model_kwargs = regression_model_kwargs

        # Store parameters for determining stop conditions
        self.max_iters = max_iters
        self.validation_history_len = validation_history_len
        self.validation_history_tol = validation_history_tol

        # Input validation for level set time step parameter
        if step is None:
            self.step = step
        else:  # Non-None => Non-automagic step => cast to float
            try:
                self.step = float(step)
            except (ValueError, TypeError):  # TypeError handles None
                msg = "`step` must be numeric or None (latter implies auto)"
                raise ValueError(msg)

        # Create the manager for the datasets
        self.datasets_handler = DatasetsHandler(
            h5_file=data_filename, imgs=imgs, segs=segs, dx=dx,
            normalize_imgs_on_convert=normalize_imgs_on_convert)

        # Split the examples into corresponding datasets
        self.datasets_handler.assign_examples_to_datasets(
            training=datasets_split[0],
            validation=datasets_split[1],
            testing=datasets_split[2],
            subset_size=subset_size,
            random_state=random_state)

        # Initialize the dictionary holding the scores
        self.scores = {
            example.key: []
            for example in self.datasets_handler.iterate_examples()
        }

        # Initialize temp data handler for managing per-iteration level set
        # values, etc.
        self.temp_data_handler = TemporaryDataHandler(tmp_dir=temp_data_dir)
        self.temp_data_handler.make_tmp_location()

    def get_seed(self, example):
        if callable(self.seeds):
            return self.seeds(example)
        else:
            return self.seeds[example.index]

    def initialize_level_sets(self):
        """ Initialize the level sets and store their values in the temp data
        hdf5 file. Also compute the automatic step size if specified
        """

        # Initialize the auto-computed step estimate
        step = numpy.inf

        with self.temp_data_handler.open_h5_file(lock=True) as temp_file:
            for example in self.datasets_handler.iterate_examples():

                msg = "Initializing level set {} / {}"
                msg = msg.format(example.index+1,
                                 self.datasets_handler.n_examples)
                logger.info(msg)

                # Create dataset group if it doesn't exist.
                group = temp_file.create_group(example.key)

                seed = self.get_seed(example)

                # Compute the initializer for this example and seed value
                u0, dist, mask = self.model.initializer(
                    img=example.img, band=self.model.band,
                    dx=example.dx, seed=seed)

                # Auto step should only use training and validation datasets
                in_train = self.datasets_handler.in_training_dataset(
                    example.key)
                in_valid = self.datasets_handler.in_validation_dataset(
                    example.key)
                if in_train or in_valid:
                    # Compute the maximum velocity for the i'th example
                    mx = numpy.abs(example.dist[mask]).max()

                    # Create the candidate step size for this example.
                    tmp = numpy.min(example.dx) / mx

                    # Assign tmp to step if it is the smallest observed so far.
                    step = tmp if tmp < step else step

                # The group consists of the current "level set field" u, the
                # signed distance transform of u (only in the narrow band), and
                # the boolean mask indicating the narrow band region.
                group.create_dataset(
                    LEVEL_SET_KEY, data=u0, compression='gzip')
                group.create_dataset(
                    SIGNED_DIST_KEY, data=dist, compression='gzip')
                group.create_dataset(
                    MASK_KEY, data=mask, compression='gzip')

            if self.step is None:
                # Assign the computed step value to class attribute and log it
                self.step = step

                msg = "Computed auto step is {:.7f}"
                logger.info(msg.format(self.step))
            elif step is not None and self.step > step:
                # Warn the user that the provided step argument may be too big
                msg = "Computed step is {:.7f} but given step is {:.7f}"
                logger.warning(msg.format(step, self.step))

    def compute_and_collect_scores(self):
        """ Collect and store the scores at the current iteration
        """

        msg = "Collecting scores (iteration = {})"
        logger.info(msg.format(self.iteration))

        with self.temp_data_handler.open_h5_file() as temp_file:
            for example in self.datasets_handler.iterate_examples():

                u = temp_file[example.key][LEVEL_SET_KEY][...]
                seg = example.seg
                score = self.model.scorer(u, seg)

                self.scores[example.key].append(score)

    def fit_regression_model(self):
        """ Fit the regression model to approximate the velocity field
        for level set motion at the current iteration
        """
        pass

    def update_level_sets(self):
        """ Update all the level sets using the learned regression model
        """
        pass

    def can_exit_early(self):
        """ Returns True when the early exit condition is satisfied
        """
        iteration = self.iteration
        va_hist_len = self.validation_history_len
        va_hist_tol = self.validation_history_tol

        logger.info("Checking early exit condition...")

        if iteration >= va_hist_len-1:

            # Set up variables for linear trend fit
            x = numpy.c_[numpy.ones(va_hist_len), numpy.arange(va_hist_len)]
            # Get scores over past `va_hist_len` iters
            # (current iteration inclusive)
            scores = self.training_scores.mean(axis=1)
            y = scores[iteration+1-va_hist_len:iteration+1]

            # The slope of the best fit line.
            slope = numpy.linalg.lstsq(x, y, rcond=None)[0][1]

            msg = ("Trend in validation scores over past {:d} "
                   "iterations is {:.7f} (tolerance = {.7f})")
            logger.info(msg.format(va_hist_len, slope, va_hist_tol))

            if slope < va_hist_tol:  # trend is not increasing sufficiently
                msg = "Early exit condition satisfied"
                logger.info(msg)
                return True

        logger.info("Early stop conditions not satisfied")
        return False

    def clean_up(self):
        """ Handles exit procedures, e.g., removing temp data
        """
        self.temp_data_handler.remove_tmp_data()
        self.model._is_fitted = True
        logger.error("FIXME: truncate models based on validation scores")
