import logging
import os
import pickle

import numpy

from level_set_machine_learning.core.fit_job_handler import FitJobHandler
from level_set_machine_learning.core.exception import ModelNotFit
from level_set_machine_learning.feature.feature_map import FeatureMap
from level_set_machine_learning.gradient import masked_gradient as mg
from level_set_machine_learning.initializer.initializer_base import (
    InitializerBase)
from level_set_machine_learning.initializer.seed import center_of_mass_seeder
from level_set_machine_learning.score_functions import jaccard
from level_set_machine_learning.util.distance_transform import (
    distance_transform)


logger = logging.getLogger(__name__.replace('level_set_machine_learning', ''))

DEFAULT_MODEL_FILENAME = 'LSML-model.pkl'


class LevelSetMachineLearning:

    def __init__(self, features, initializer, scorer=jaccard, band=3,
                 normalize_imgs=True):
        """
        Initialize a level set machine learning object

        Note
        ----
        The parameters required at initialization are those that are necessary
        both at train- and run-time.

        Parameters
        ----------
        features: List(BaseFeature)
            A list of image or shape features that derive from BaseImageFeature
            or BaseShapeFeature

        initializer: class
            Provides the initial segmentation guess, given an image.
            A subclass of
            :class:`level_set_machine_learning.initializer.InitializerBase`

        scorer: function, default=jaccard
            Has signature::

                scorer(u, seg)

            where `u` is a level set iterate and `seg` is the ground-truth
            segmentation. The default (None) uses the jaccard overlap
            between `u > 0` and `seg`.

        band: float, default=3
            The "narrow band" distance.

        normalize_imgs: bool, default=True
            If True, then the provided images are individually normalized
            by their means and standard deviations


        """
        # Create the feature map comprising the given features
        self.feature_map = FeatureMap(features=features)

        # Validate the level set initializer
        if not isinstance(initializer, InitializerBase):
            msg = ("`initializer` should be a class derived from "
                   "`level_set_machine_learning.initializer.InitializerBase`")
            raise ValueError(msg)
        self.initializer = initializer

        self.scorer = scorer
        self.band = band
        self.normalize_imgs = normalize_imgs

        # These are filled in with `DatasetProxy` post fit
        self.training_data = None
        self.validation_data = None
        self.testing_data = None

        self.fit_job_handler = None
        self._is_fitted = False

    def fit(self,
            # Args
            data_filename,
            regression_model_class,
            regression_model_kwargs,
            # KWArgs
            balance_regression_targets=True,
            datasets_split=(0.6, 0.2, 0.2),
            dx=None,
            imgs=None,
            max_iters=100,
            random_state=numpy.random.RandomState(),
            save_filename=DEFAULT_MODEL_FILENAME,
            seeds=center_of_mass_seeder,
            segs=None,
            step=None,
            subset_size=None,
            temp_data_dir=os.path.curdir,
            validation_history_len=5,
            validation_history_tol=0.0,
            ):
        """ Fit a level set machine learning segmentation model

        Parameters
        ----------
        data_filename: str
            The filename of the hdf5 file of converted data or the filename
            of an existing hdf5 file with required structure. If the
            file `h5_file` does not exist, then `imgs` and `segs` *must* be
            provided.

        regression_model_class: class
            The regression model class. At each iteration of level set
            evolution, a new instance of this class is instantiated to create
            the respective velocity model for the iteration

        regression_model_kwargs: dict
            A dictionary of keyword arguments used to instantiate the provided
            :code:`regression_model_class`. For example,
            :code:`{'n_estimators': 100, 'random_state': RandomState(123)}`
            might be used with corresponding :code:`RandomForestRegressor`
            model class.

        balance_regression_targets: bool, default=True
            If True (default), then the target arrays for regression
            are balanced by randomly down-sampling to contain equal negative
            and positive portions

        imgs: list of ndarray, default=None
            The list of images to be used if :code:`data_file` points to a
            (currently) non-existent hdf5 file. The default of None
            assumes that `data_file` points to an existing hdf5 file

        segs: list of ndarray
            The list of segmentation to be used if :code:`data_file` points to
            a (currently) non-existent hdf5 file. The default of None
            assumes that `data_file` points to an existing hdf5 file

        dx: list of ndarray, default=None
            The list of respective delta terms for the provided images.
            :code:`dx[i]` should be a list or array corresponding to the
            i'th image with length corresponding to the image dimensions.
            The default of None assumes isotropicity with a value of 1.

        seeds: list or callable, default=center_of_mass_seeder
            A list of seed points for each image example. Each respective
            seed point is passed to the :code:`initializer` function.
            Alternatively, a callable can be passed with signature,
            :code:`seeds(dataset_example)`, where the argument is an instance
            of :code:`DatasetExample` from
            :module:`level_set_machine_learning.core.datasets_handler`.

        save_filename: str
            The filename to where the fitted level set machine learning model
            will be pickled

        datasets_split: 3-tuple, default=(0.6, 0.2, 0.2)
            The items in the tuple correspond to the training, validation, and
            testing datasets. If each item is a float, then the provided data
            is randomly split with probabilities of respective values.
            Alternatively, each item in the 3-tuple may be a list of integer
            indexes explicitly prescribing each example to a specific dataset
            for fitting.

        subset_size: int, default=None
            If datasets are randomly partitioned, then the full dataset
            is first down-sampled to be `subset_size` before partitioning

        step: float, default=None
            The step size for updating the level sets, i.e., the "delta t"
            term in the discretization of :math:`u_t = \\nu \\| Du \\|`.

            The default None determines the step size automatically
            by using the reciprocal of the maximum ground truth speed values
            in the narrow band of the level sets in the training data at
            the first iterate (i.e., at initializer). The CFL condition
            is satisfied by taking the maximum speed over ALL spatial
            coordinates, but we attempt to avoid this prohibitively
            small step size by assuming that the maximum speed observed
            will be in the first iteration (i.e., that u0 is farthest
            from the ground-truth).

        temp_data_dir: str, default=os.path.curdir
            Where to store the temporary data that is created during the
            fitting process. This data is removed after fitting and includes,
            for example, the per-iteration level set values

        validation_history_len: int, default=5
            The number of past iterations back from the current to check
            scores over the validation dataset to monitor progress from

        validation_history_tol: float, default=0.0
            If the linear trend in scores over the last
            :code:`validation_history_len` iterations is less than this
            value, then we exit early.

        max_iters: int, default=100
            The fixed maximum number of iterations

        random_state: numpy.random.RandomState
            Provide for reproducible results.

        """
        # Dirty tricks to initialize the fit handler...
        kwargs = locals()
        kwargs['model'] = kwargs.pop('self')
        self.fit_job_handler = FitJobHandler(**kwargs)

        # Set up the level sets according the initializer functions
        self.fit_job_handler.initialize_level_sets()

        # Compute and store scores at initialization (iteration = 0)
        self.fit_job_handler.compute_and_collect_scores()

        for self.fit_job_handler.iteration in range(1, max_iters+1):
            logger.info("*"*40)
            msg = "Beginning iteration {:03d}"
            logger.info(msg.format(self.fit_job_handler.iteration))
            logger.info("*"*40)

            self.fit_job_handler.fit_regression_model()
            self.fit_job_handler.update_level_sets()
            self.fit_job_handler.compute_and_collect_scores()
            self.save(save_filename)

            if self.fit_job_handler.can_exit_early():
                break

        # Handle fit related exit tasks
        self.fit_job_handler.clean_up()

        # Write the model to disk
        self.save(save_filename)

    def save(self, filename):
        """ Write the LevelSetMachineLearning to disk
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Load a LevelSetMachineLearning model from disk
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        return model

    #################################################################
    # Attributes / methods available after model fit
    #################################################################

    def _requires_fit(method):
        """ Decorator for methods that require a fitted model
        """
        def method_wrapped(self, *args, **kwargs):
            if not self._is_fitted:
                raise ModelNotFit("This model has not been fit yet")

            return method(self, *args, **kwargs)

        return method_wrapped

    @_requires_fit
    def segment(self, img, dx=None, verbose=True, on_iterate=None,
                iterate_until_validation_max=True):
        """
        Segment `img`.

        Parameters
        ----------
        img: ndarray
            The image

        dx: ndarray or list, default=None
            List of the spatial delta terms along each axis; default is ones

        verbose: bool, default=True
            Print progress

        on_iterate: callable or list of callables, default=None
            Supply a callable to be performed prior to each level set iteration
            The expected signature is :code:`on_iterate(i, u)`
            where :code:`u` is level set function at iteration :code:`i`

        iterate_until_validation_max: bool, default=True
            If True, then the iteration will stop at the index corresponding
            to the global max over the validation data; otherwise, iterations
            continue for all possible, i.e., for the number regression models
            that exist

        Returns
        -------
        us: ndarray
            `us` is shape `(len(self.models)+1,) + img.shape`, where `us[i]`
            is the i'th iterate of the level set function and `us[i] > 0`
            yields an approximate boolean-mask segmentation of `img` at
            the i'th iteration

        """
        ############################################################
        # Input validation
        if self.normalize_imgs:
            img_ = (img - img.mean()) / img.std()
        else:
            img_ = img

        if iterate_until_validation_max:
            n_iters = self.validation_scores.mean(axis=1).argmax()
        else:
            n_iters = len(self.fit_job_handler.iteration)

        dx = numpy.ones(img_.ndim) if dx is None else dx

        if dx.shape[0] != img_.ndim:
            raise ValueError("`dx` has incorrect number of elements.")

        if on_iterate:
            if not isinstance(on_iterate, list):
                on_iterate = [on_iterate]

            if not all([callable(func) for func in on_iterate]):
                msg = "All on_iterate items must be callable"
                raise TypeError(msg)
        # End Input validation
        ############################################################

        us = numpy.zeros((n_iters+1,) + img.shape)
        us[0], dist, mask = self.initializer(img_, self.band, dx=dx)

        if on_iterate:
            for func in on_iterate:
                func(0, us[0])

        velocity = numpy.zeros(img.shape)

        print_string = "Iter: {:02d}"
        if verbose:
            print(print_string.format(0))

        for i in range(n_iters):
            us[i+1] = us[i].copy()

            if mask.any():
                # Compute the features, and use the model to predict velocity
                features = self.feature_map(
                    u=us[i], img=img_, dist=dist, mask=mask, dx=dx)

                velocity[mask] = self._get_regression_model_prediction(
                    iteration=i, features=features[mask])

                gmag = mg.gradient_magnitude_osher_sethian(
                    arr=us[i], nu=velocity, mask=mask, dx=dx)

                # Update the level set.
                us[i+1][mask] += self.step*velocity[mask]*gmag[mask]

                # Check for level set vanishing.
                dist, mask = distance_transform(
                    arr=us[i+1], band=self.band, dx=dx)

            if verbose:
                print(print_string.format(i+1))

            if on_iterate:
                for func in on_iterate:
                    func(i+1, us[i+1])

        return us

    @_requires_fit
    def _get_regression_model_prediction(self, iteration, features):
        regression_model = self.fit_job_handler._load_regression_model(
            iteration=iteration)
        velocity = regression_model.predict(features)
        return velocity

    @property
    @_requires_fit
    def step(self):
        return self.fit_job_handler.step

    @_requires_fit
    def _get_scores_for_dataset(self, dataset_key):
        """ Get an array of scores, shape `(n_iterations, n_examples)`
        """
        return numpy.array([
            self.fit_job_handler.scores[example]
            for example in self.fit_job_handler.datasets_handler.datasets[dataset_key]  # noqa
        ]).T  # <= transpose to get desired shape

    @property
    def training_scores(self):
        from .datasets_handler import TRAINING_DATASET_KEY
        return self._get_scores_for_dataset(TRAINING_DATASET_KEY)

    @property
    def validation_scores(self):
        from .datasets_handler import VALIDATION_DATASET_KEY
        return self._get_scores_for_dataset(VALIDATION_DATASET_KEY)

    @property
    def testing_scores(self):
        from .datasets_handler import TESTING_DATASET_KEY
        return self._get_scores_for_dataset(TESTING_DATASET_KEY)

    @_requires_fit
    def regression_model(self, iteration):
        return self.fit_job_handler._load_regression_model(iteration=iteration)
