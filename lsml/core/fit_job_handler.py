import logging
import multiprocessing
import os
import pickle
import sys

import numpy

from .datasets_handler import DatasetsHandler
from .exception import ModelAlreadyFit
from .temporary_data_handler import (
    LEVEL_SET_KEY, MASK_KEY, SIGNED_DIST_KEY, TemporaryDataHandler,)
from lsml.gradient import masked_gradient
from lsml.util.balance_mask import balance_mask
from lsml.util.distance_transform import (
    distance_transform)


_logger_name = __name__.rsplit('.', 1)[-1]
logger = logging.getLogger(_logger_name)

REGRESSION_MODEL_DIRNAME = 'regression-models'
REGRESSION_MODEL_FILENAME = 'regression-model-{:d}.pkl'


def setup_logging(capture_std_out=True):
    """ Sets up logging formatting, etc

    Parameters
    ----------
    capture_std_out : bool, default=True
        If True, then std out gets captured and written to the log
        file. This is useful for capture the printed output from
        models for which you do not have control to redirect their
        status outputs (e.g., the fitting status of a scikit-learn model)
    """
    logger_filename = "fit-log.txt"

    if os.path.exists(logger_filename):
        os.remove(logger_filename)

    line_fmt = ("[%(asctime)s] [%(name)s:%(lineno)d] "
                "%(levelname)-8s %(message)s")

    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        filename=logger_filename, format=line_fmt,
        datefmt=date_fmt, level=logging.DEBUG)

    if capture_std_out:
        class StdOutLogger:
            _logger = logging.getLogger('STD OUT CAPTURE')

            def write(self, message):
                if message != '\n':
                    self._logger.info(message)

            def flush(self):
                pass

        sys._stdout = sys.stdout
        sys.stdout = StdOutLogger()


class FitJobHandler:
    """ Manages attributes and model fitting data/procedures
    """
    def __init__(self,
                 balance_regression_targets,
                 data_filename,
                 datasets_split,
                 dx,
                 imgs,
                 max_iters,
                 model,
                 random_state,
                 regression_model_class,
                 regression_model_kwargs,
                 save_filename,
                 seeds,
                 segs,
                 step,
                 subset_size,
                 temp_data_dir,
                 validation_history_len,
                 validation_history_tol,
                 redirect_stdout_to_logfile,
                 ):
        """
        See :class:`level_set_machine_learning.LevelSetMachineLearning`
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

        # Set up logging formatting, file location, etc.
        setup_logging(capture_std_out=redirect_stdout_to_logfile)

        # Store the seeds list or function
        self.seeds = seeds

        # Store the RandomState instance for re-used
        # on balancing masks
        self.random_state = random_state

        # Store the balancing option flag
        self.balance_regression_targets = balance_regression_targets

        # The LevelSetMachineLearning instance
        self.model = model

        # Initialize the regression models to empty list
        self.regression_models = []

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
            h5_file=data_filename, imgs=imgs, segs=segs, dx=dx)

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

    def _log_with_iter(self, msg, level='info'):
        """ Write to the logger with the current iteration number prepended
        to the log message
        """

        full_message = "(Iteration = {:03d}) {:s}".format(self.iteration, msg)

        if level == 'info':
            logger.info(full_message)
        elif level == 'debug':
            logger.debug(full_message)
        elif level == 'warning':
            logger.warning(full_message)
        elif level == 'error':
            logger.error(full_message)
        else:
            raise ValueError("Unknown log level: {}".format(level))

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

                # Normalize the image if necessary
                if self.model.normalize_imgs:
                    img_ = (example.img -
                            example.img.mean()) / example.img.std()
                else:
                    img_ = example.img

                # Compute the initializer for this example and seed value
                u0, dist, mask = self.model.initializer(
                    img=img_, band=self.model.band,
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
        self._log_with_iter("Collecting scores")

        with self.temp_data_handler.open_h5_file() as temp_file:
            for example in self.datasets_handler.iterate_examples():

                u = temp_file[example.key][LEVEL_SET_KEY][...]
                seg = example.seg
                score = self.model.scorer(u, seg)

                self.scores[example.key].append(score)

        ################################################
        # Report scores via the logger

        from lsml.core.datasets_handler import (
            TRAINING_DATASET_KEY, VALIDATION_DATASET_KEY, TESTING_DATASET_KEY)

        dataset_keys = (
            TRAINING_DATASET_KEY, VALIDATION_DATASET_KEY, TESTING_DATASET_KEY)

        for dataset_key in dataset_keys:
            mean = numpy.mean([
                self.scores[key][-1]
                for key in self.datasets_handler.iterate_keys(dataset_key)
            ])
            msg = "Average score over {:s} = {:.7f}"
            self._log_with_iter(msg.format(dataset_key, mean))

    def _featurize_all_images(self, dataset_key):
        """ Featurize all the images in the dataset given by the argument

        dataset: str
            The key for the dataset for which we will featurize the images

        Returns
        -------
        features, targets: ndarray (n_examples, n_features) and (n_examples,)
            features is a matrix storing the feature vector in each row,
            whereas targets is a vector storing the corresponding target
            value for each feature vector (i.e., the ground-truth signed
            distance value at the spatial coordinate for which the
            feature vector was computed).

        """

        ###########################################################
        # Precompute total number of feature vector examples

        # Count the number of points collected
        count = 0

        if self.balance_regression_targets:
            bal_masks = []

        examples = self.datasets_handler.iterate_examples(
            dataset_key=dataset_key)

        for example in examples:

            with self.temp_data_handler.open_h5_file() as temp_file:
                mask = temp_file[example.key][MASK_KEY][...]

            if self.balance_regression_targets:

                bal_mask = balance_mask(
                    example.dist[mask], random_state=self.random_state)
                count += bal_mask.sum()
                bal_masks.append(bal_mask)

            else:
                count += mask.sum()

        features = numpy.zeros((count, self.model.feature_map.n_features))
        targets = numpy.zeros((count,))

        index = 0

        ##################################################################
        # Compute the feature vectors and place them in the feature matrix

        examples = self.datasets_handler.iterate_examples(
            dataset_key=dataset_key)

        for i, example in enumerate(examples):

            if self.model.normalize_imgs:
                img_ = (example.img - example.img.mean()) / example.img.std()
            else:
                img_ = example.img

            with self.temp_data_handler.open_h5_file() as tf:
                u = tf[example.key][LEVEL_SET_KEY][...]
                dist = tf[example.key][SIGNED_DIST_KEY][...]
                mask = tf[example.key][MASK_KEY][...]

            if not mask.any():
                continue  # Otherwise, repeat the loop until a non-empty mask

            # Compute features.
            features_current = self.model.feature_map(
                u=u, img=img_, dist=dist, mask=mask, dx=example.dx)

            if self.balance_regression_targets:
                bmask = bal_masks[i]
                next_index = index + example.dist[mask][bmask].shape[0]
                features[index:next_index] = features_current[mask][bmask]
                targets[index:next_index] = example.dist[mask][bmask]
            else:
                next_index = index + mask.sum()
                features[index:next_index] = features_current[mask]
                targets[index:next_index] = example.dist[mask]

            index = next_index

        return features, targets

    def fit_regression_model(self):
        """ Fit the regression model to approximate the velocity field
        for level set motion at the current iteration
        """
        from lsml.core.datasets_handler import (
            TRAINING_DATASET_KEY)

        self._log_with_iter("Fitting regression model")

        # Get input and output variables
        features, targets = self._featurize_all_images(
            dataset_key=TRAINING_DATASET_KEY)

        msg = "... regression training set shapes: features={}, targets={}"
        self._log_with_iter(msg.format(features.shape, targets.shape))

        self.temp_data_handler.store_array('features.npy', features)
        self.temp_data_handler.store_array('targets.npy', targets)

        fit_proc = multiprocessing.Process(target=self._fit_regression_model)
        fit_proc.start()
        fit_proc.join()

        if fit_proc.exitcode != 0:
            msg = ("Exiting due to error during regression model fit; temp "
                   "data remains intact at {}").format(
                self.temp_data_handler.tmp_data_location)

            self._log_with_iter(msg, level='error')
            raise RuntimeError(msg)

    def _fit_regression_model(self):
        """ Target for fitting the regression model in a new process
        """

        try:

            features = self.temp_data_handler.load_array('features.npy')
            targets = self.temp_data_handler.load_array('targets.npy')

            # Instantiate the regression model
            regression_model = self.regression_model_class(
                **self.regression_model_kwargs)

            # Fit it!
            regression_model.fit(features, targets)

            self._store_regression_model(regression_model)

        except Exception as e:
            import traceback
            msg = "Error occurred during regression model fit:\n{}"
            msg = msg.format(traceback.format_exc())
            self._log_with_iter(msg, level='error')
            raise e

    def _store_regression_model(self, regression_model, iteration=None):
        """ Pickle the regression model for given iteration to disk. None
        uses `self.iteration`.
        """
        iter = iteration or self.iteration

        # Create the folder for storing the regression models if necessary
        if not os.path.exists(REGRESSION_MODEL_DIRNAME):
            os.makedirs(REGRESSION_MODEL_DIRNAME)

        # Build the regression model path from defaults and current iteration
        regression_model_filename = REGRESSION_MODEL_FILENAME.format(iter)
        regression_model_path = os.path.join(REGRESSION_MODEL_DIRNAME,
                                             regression_model_filename)

        # Pickle it!
        with open(regression_model_path, 'wb') as f:
            pickle.dump(regression_model, f)

    def _load_regression_model(self, iteration=None):
        """ Load the regression model for the given iteration from disk. The
        default of None uses `self.iteration`.
        """
        if iteration is None:
            iter = self.iteration
        else:
            iter = iteration

        # Build the regression model path from defaults and current iteration
        regression_model_filename = REGRESSION_MODEL_FILENAME.format(iter)
        regression_model_path = os.path.join(REGRESSION_MODEL_DIRNAME,
                                             regression_model_filename)

        # Un-pickle it!
        with open(regression_model_path, 'rb') as f:
            regression_model = pickle.load(f)

        return regression_model

    def update_level_sets(self):
        """ Update all the level sets using the learned regression model
        """
        self._log_with_iter("Updating level sets")

        regression_model = self._load_regression_model()

        for example in self.datasets_handler.iterate_examples():

            if self.model.normalize_imgs:
                img_ = (example.img - example.img.mean()) / example.img.std()
            else:
                img_ = example.img

            with self.temp_data_handler.open_h5_file(lock=True) as tf:
                mask = tf[example.key][MASK_KEY][...]

                # Only update if the mask is not empty
                if mask.any():
                    u = tf[example.key][LEVEL_SET_KEY][...]
                    dist = tf[example.key][SIGNED_DIST_KEY][...]

                    # Compute features.
                    features = self.model.feature_map(
                        u=u, img=img_, dist=dist, mask=mask, dx=example.dx)

                    # Compute approximate velocity from features
                    velocity = numpy.zeros_like(u)
                    velocity[mask] = regression_model.predict(features[mask])

                    # Compute gradient magnitude using upwind method
                    gmag = masked_gradient.gradient_magnitude_osher_sethian(
                        arr=u, nu=velocity, mask=mask, dx=example.dx)

                    # Here's the actual level set update.
                    u[mask] += self.step*velocity[mask]*gmag[mask]

                    # Update the distance transform and mask
                    # after the level set field has been updated
                    dist, mask = distance_transform(
                        arr=u, band=self.model.band, dx=example.dx)

                    # Update the data in the temp file
                    tf[example.key][LEVEL_SET_KEY][...] = u
                    tf[example.key][SIGNED_DIST_KEY][...] = dist
                    tf[example.key][MASK_KEY][...] = mask

    def can_exit_early(self):
        """ Returns True when the early exit condition is satisfied
        """
        from lsml.core.datasets_handler import (
            VALIDATION_DATASET_KEY)

        iteration = self.iteration
        va_hist_len = self.validation_history_len
        va_hist_tol = self.validation_history_tol

        self._log_with_iter("Checking early exit condition")

        if iteration >= va_hist_len-1:

            # Set up variables for linear trend fit
            x = numpy.c_[numpy.ones(va_hist_len), numpy.arange(va_hist_len)]

            # Get scores over past `va_hist_len` iters
            # (current iteration inclusive)
            scores = numpy.array([
                self.scores[example_key]
                for example_key in self.datasets_handler.iterate_keys(
                    dataset_key=VALIDATION_DATASET_KEY)
            ]).mean(axis=0)

            y = scores[iteration+1-va_hist_len:iteration+1]

            # The slope of the best fit line.
            slope = numpy.linalg.lstsq(x, y, rcond=None)[0][1]

            msg = "Validation trend = {:.6f} (Exits when ≤ {:.6f})"
            self._log_with_iter(msg.format(slope, va_hist_tol))

            if slope < va_hist_tol:  # trend is not increasing sufficiently
                msg = "Early exit condition satisfied"
                self._log_with_iter(msg)
                return True

        self._log_with_iter("Early stop conditions not satisfied")
        return False

    def clean_up(self):
        """ Handles exit procedures, e.g., removing temp data
        """
        from lsml.core.datasets_handler import (
            DatasetProxy,
            TESTING_DATASET_KEY, TRAINING_DATASET_KEY, VALIDATION_DATASET_KEY)

        # Eliminate the temporary data used during fit
        self.temp_data_handler.remove_tmp_data()

        # Build the dataset proxy objects and attach to the model
        self.model.training_data = DatasetProxy(
            datasets_handler=self.datasets_handler,
            dataset_key=TRAINING_DATASET_KEY)

        self.model.validation_data = DatasetProxy(
            datasets_handler=self.datasets_handler,
            dataset_key=VALIDATION_DATASET_KEY)

        self.model.testing_data = DatasetProxy(
            datasets_handler=self.datasets_handler,
            dataset_key=TESTING_DATASET_KEY)

        # Give stdout back
        try:
            sys.stdout = sys._stdout
        except AttributeError:
            pass

        self.model._is_fitted = True
