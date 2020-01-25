from collections import namedtuple
import contextlib
import logging
import os

import h5py
import numpy
import skfmm


_logger_name = __name__.rsplit('.', 1)[-1]
logger = logging.getLogger(_logger_name)


TRAINING_DATASET_KEY = 'training'
VALIDATION_DATASET_KEY = 'validation'
TESTING_DATASET_KEY = 'testing'
DATASET_KEYS = (
    TRAINING_DATASET_KEY,
    VALIDATION_DATASET_KEY,
    TESTING_DATASET_KEY
)
EXAMPLE_KEY = "example-{:d}"
IMAGE_KEY = "image"
SEGMENTATION_KEY = "segmentation"
DISTANCE_TRANSFORM_KEY = "distance-transform"


# Yielded by dataset manager generators
DatasetExample = namedtuple(
    'DatasetExample',
    ['index', 'key', 'img', 'seg', 'dist', 'dx'])


class DatasetsHandler:
    """ Handles internal dataset operations during model fitting
    """

    def __init__(self, h5_file, imgs=None, segs=None, dx=None, compress=True):
        """ Initialize a dataset manager

        Parameters
        ----------
        h5_file: str
            A (possibly already existing) hdf5 dataset

        imgs: List(ndarray), default=None
            List of images. Necessary when `h5_file` doe not exist.

        segs: List(ndarray), default=None
            List of respective segmentations for :code:`imgs`. Necessary when
            `h5_file` doe not exist.

        dx: List(ndarray), default=None
            The delta spacing along the axis directions in the provided
            images and segmentations. Used when `h5_file` doesn't exist. In
            that case, the default of None uses all ones.

        compress: bool, default=True
            When the image and segmentation data are stored in the hdf5 file
            this flag indicates whether or not to use compression.

        Note
        ----
        Either :code:`h5_file` should be the name of an existing h5 file with
        appropriate structure (see :method:`convert_to_hdf5`) or `imgs`
        and `segs` should be non-None, and the hdf5 file will creating
        using the provided images and segmentations and we be named with the
        argument `h5_file`.

        """
        self.h5_file = os.path.abspath(h5_file)
        self.datasets = {
            dataset_key: [] for dataset_key in self._iterate_dataset_keys()}

        if not os.path.exists(self.h5_file):
            if imgs is None or segs is None:
                msg = ("Provided `h5_file` {} doesn't exist but no image or "
                       "segmentation data provided")
                raise ValueError(msg.format(h5_file))

            # Perform the conversion to hdf5
            self.convert_to_hdf5(
                imgs=imgs, segs=segs, dx=dx, compress=compress)

        with h5py.File(self.h5_file, mode='r') as hf:
            self.n_examples = len(hf.keys())

    def convert_to_hdf5(self, imgs, segs, dx=None, compress=True):
        """ Convert a dataset of images and boolean segmentations
        to hdf5 format, which is required for the level set routine.

        The format assuming `hf` is and h5py `File` is as follows::

            'i'
            |_ img
            |_ seg
            |_ dist
            |_ attrs
               |_ dx

        Parameters
        ----------
        imgs: list of ndarray
            The list of image examples for the dataset

        segs: list of ndarray
            The list of image examples for the dataset

        dx: list of ndarray, shape=(n_examples, img.ndim), default=None
            The resolutions along each axis for each image. The default (None)
            assumes the resolution is 1 along each axis direction, but this
            might not be the case for anisotropic data.

        compress: bool, default=True
            If True, :code:`gzip` compression with default compression
            options (level=4) is used for the images and segmentations.

        """
        # Check if the file already exists and abort if so.
        if os.path.exists(self.h5_file):
            msg = "Dataset already exists at {}"
            raise FileExistsError(msg.format(self.h5_file))

        # Setup some variables
        n_examples = len(imgs)
        ndim = imgs[0].ndim
        compress_method = "gzip" if compress else None

        ######################
        # Input validation

        if len(imgs) != len(segs):
            msg = "Mismatch in number of examples: imgs ({}), segs ({})"
            raise ValueError(msg.format(len(imgs), len(segs)))

        for i in range(n_examples):
            img = imgs[i]
            seg = segs[i]

            # Validate image data type
            if img.dtype != numpy.float:
                msg = "imgs[{}] (dtype {}) was not float"
                raise TypeError(msg.format(i, img.dtype))

            # Validate segmentation data type
            if seg.dtype != numpy.bool:
                msg = "seg[{}] (dtype {}) was not bool"
                raise TypeError(msg.format(i, seg.dtype))

            if img.ndim != ndim:
                msg = "imgs[{}] (ndim={}) did not have correct dimensions ({})"
                raise ValueError(msg.format(i, img.ndim, ndim))

            if seg.ndim != ndim:
                msg = "segs[{}] (ndim={}) did not have correct dimensions ({})"
                raise ValueError(msg.format(i, seg.ndim, ndim))

            if img.shape != seg.shape:
                msg = "imgs[{}] shape {} does not match segs[{}] shape {}"
                raise ValueError(msg.format(i, img.shape, i, seg.shape))

        # Check dx if provided and is correct shape.
        if dx is None:
            dx = numpy.ones((n_examples, ndim), dtype=numpy.float)
        else:
            if dx.shape != (n_examples, ndim):
                msg = "`dx` was shape {} but should be shape {}"
                raise ValueError(msg.format(dx.shape, (n_examples, ndim)))

        # End input validation
        ##########################

        hf = h5py.File(self.h5_file, mode='w')

        for i in range(n_examples):

            msg = "Creating dataset entry {} / {}"
            logger.info(msg.format(i+1, n_examples))

            # Create a group for the i'th example
            g = hf.create_group(EXAMPLE_KEY.format(i))

            # Store the i'th image
            g.create_dataset(IMAGE_KEY,
                             data=imgs[i], compression=compress_method)

            # Store the i'th segmentation
            g.create_dataset(SEGMENTATION_KEY,
                             data=segs[i], compression=compress_method)

            # Compute the signed distance transform of the ground-truth
            # segmentation and store it.
            dist = skfmm.distance(2*segs[i].astype(numpy.float)-1, dx=dx[i])
            g.create_dataset(DISTANCE_TRANSFORM_KEY,
                             data=dist, compression=compress_method)

            # Store the delta terms as an attribute.
            g.attrs['dx'] = dx[i]

        # Close up shop
        hf.close()

    def assign_examples_to_datasets(
            self, training, validation, testing, subset_size, random_state):
        """ Assign the dataset example keys to training, validation,
        or testing

        training: float, or list of int
            A probability value or a list of indices of examples that belong
            to the training dataset

        validation: float, or list of int
            A probability value or a list of indices of examples that belong
            to the validation dataset

        testing: float, or list of int
            A probability value or a list of indices of examples that belong
            to the testing dataset

        subset_size: int or None
            If datasets are randomly partitioned, then the full dataset
            is first down-sampled to be `subset_size` before partitioning

        random_state: numpy.random.RandomState, default=None
            The random state is used only to perform the randomized split
            into when training/validation/testing are provided as probability
            values

        """
        if not random_state:
            random_state = numpy.random.RandomState()
            msg = ("RandomState not provided; results will "
                   "not be reproducible")
            logger.warning(msg)
        elif not isinstance(random_state, numpy.random.RandomState):
            msg = "`random_state` ({}) not instance numpy.random.RandomState"
            raise TypeError(msg.format(type(random_state)))

        if all([isinstance(item, float)
                for item in (training, validation, testing)]):
            # Random split
            self.assign_examples_randomly(
                probabilities=(training, validation, testing),
                subset_size=subset_size,
                random_state=random_state)
        elif all([
            (isinstance(index_list, list) and
             [isinstance(index, int) for index in index_list])
            for index_list in (training, validation, testing)
        ]):
            # Each is list of ints
            self.assign_examples_by_indices(
                training_dataset_indices=training,
                validation_dataset_indices=validation,
                testing_dataset_indices=testing)
        else:
            # Bad values supplied
            msg = ("`training`, `validation`, and `testing` should be "
                   "all floats or all list of ints")
            raise ValueError(msg)

    def assign_examples_by_indices(self,
                                   training_dataset_indices,
                                   validation_dataset_indices,
                                   testing_dataset_indices):
        """ Specify which of the data should belong to training, validation,
        and testing datasets. Automatic randomization is possible: see keyword
        argument parameters.

        Parameters
        ----------
        training_dataset_indices: list of integers
            The list of indices of examples that belong to the training dataset

        validation_dataset_indices: list of integers
            The list of indices of examples that belong to the validation
            dataset

        testing_dataset_indices: list of integers
            The list of indices of examples that belong to the testing dataset

        """
        if not all([isinstance(index, int)
                    for index in training_dataset_indices]):
            msg = "Training data indices must be a list of integers"
            raise ValueError(msg)

        if not all([isinstance(index, int)
                    for index in validation_dataset_indices]):
            msg = "Validation data indices must be a list of integers"
            raise ValueError(msg)

        if not all([isinstance(index, int)
                    for index in testing_dataset_indices]):
            msg = "Training data indices must be a list of integers"
            raise ValueError(msg)

        self.datasets[TRAINING_DATASET_KEY] = [
            self._example_key_from_index(index)
            for index in training_dataset_indices
        ]
        self.datasets[VALIDATION_DATASET_KEY] = [
            self._example_key_from_index(index)
            for index in validation_dataset_indices
        ]
        self.datasets[TESTING_DATASET_KEY] = [
            self._example_key_from_index(index)
            for index in testing_dataset_indices
        ]

    def assign_examples_randomly(self, probabilities,
                                 subset_size, random_state):
        """ Assign examples randomly into training, validation, and testing

        Parameters
        ----------
        probabilities: 3-tuple of floats
            The probability of being placed in the training, validation
            or testing

        subset_size: int
            If provided, then should be less than or equal to
            :code:`len(keys)`. If given, then :code:`keys` is first
            sub-sampled by :code:`subset_size`
            before splitting.

        random_state: numpy.random.RandomState
            Provide for reproducible results

        """
        with self.open_h5_file() as hf:
            keys = list(hf.keys())

        if subset_size is not None and subset_size > len(keys):
            raise ValueError("`subset_size` must be <= `len(keys)`")

        if subset_size is None:
            subset_size = len(keys)

        sub_keys = random_state.choice(keys, replace=False, size=subset_size)
        n_keys = len(sub_keys)

        # This generates a matrix size `(n_keys, 3)` where each row
        # is an indicator vector indicating to which dataset the key with
        # respective row index should be placed into.
        indicators = random_state.multinomial(
            n=1, pvals=probabilities, size=n_keys)

        # Get the dataset keys for iteration
        dataset_keys = self._iterate_dataset_keys()

        for idataset_key, dataset_key in enumerate(dataset_keys):

            # Include example indexes when the indicator array is 1
            # in the respective slot for the given dataset index in the
            # outer for loop
            self.datasets[dataset_key] = [
                self._example_key_from_index(index)
                for index, indicator in enumerate(indicators)
                if list(indicator).index(1) == idataset_key
            ]

    @contextlib.contextmanager
    def open_h5_file(self):
        """ Opens the data file
        """
        h5 = None
        try:
            h5 = h5py.File(self.h5_file, mode='r')
            yield h5
        finally:
            if h5:
                h5.close()

    def _iterate_dataset_keys(self):
        """ Iterates through the dataset keys
        """
        for dataset_key in DATASET_KEYS:
            yield dataset_key

    def _example_key_from_index(self, index):
        """ Get the example key for the corresponding index
        """
        return EXAMPLE_KEY.format(index)

    def get_dataset_for_example_key(self, example_key):
        """ Get the dataset for the corresponding example key

        Returns
        -------
        dataset_key: str or None
            One of TRAINING_DATASET_KEY, VALIDATION_DATASET_KEY, or
            TESTING_DATASET_KEY if found; otherwise, returns None.

        """
        if self.in_training_dataset(example_key):
            return TRAINING_DATASET_KEY
        elif self.in_validation_dataset(example_key):
            return VALIDATION_DATASET_KEY
        elif self.in_testing_dataset(example_key):
            return TESTING_DATASET_KEY
        else:
            return None

    def iterate_keys(self, dataset_key=None):

        for i in range(self.n_examples):

            example_key = self._example_key_from_index(i)

            if (not dataset_key or self._in_dataset(
                    example_key=example_key, dataset_key=dataset_key)):
                yield example_key

    def get_example_by_index(self, index):
        """ Get the `DatasetExample` corresponding to `index`
        """
        with self.open_h5_file() as hf:

            example_key = self._example_key_from_index(index)

            example = DatasetExample(
                index=index,
                key=example_key,
                img=hf[example_key][IMAGE_KEY][...],
                seg=hf[example_key][SEGMENTATION_KEY][...],
                dist=hf[example_key][DISTANCE_TRANSFORM_KEY][...],
                dx=hf[example_key].attrs['dx']
            )

        return example

    def _get_example_by_key(self, example_key):

        with self.open_h5_file() as hf:

            example = DatasetExample(
                index=int(example_key.split('-')[-1]),
                key=example_key,
                img=hf[example_key][IMAGE_KEY][...],
                seg=hf[example_key][SEGMENTATION_KEY][...],
                dist=hf[example_key][DISTANCE_TRANSFORM_KEY][...],
                dx=hf[example_key].attrs['dx']
            )

        return example

    def iterate_examples(self, dataset_key=None):
        """ Iterates through the hdf5 dataset

        Parameters
        ----------
        dataset_key: str, default=None
            Limit the iterations to the given dataset; None yields all examples

        Returns
        -------
        dataset: generator
            The return generator returns
            `(i, key, img[i], seg[i], dist[i], dx[i])`
            at each iteration, where i is the index and key is the
            key into the hdf5 dataset for the respective index

        """
        with self.open_h5_file() as hf:
            for i in range(self.n_examples):

                example_key = self._example_key_from_index(i)

                # Skip if the example is not in the desired dataset
                if (dataset_key and
                        not self._in_dataset(example_key, dataset_key)):
                    continue

                yield DatasetExample(
                    index=i,
                    key=example_key,
                    img=hf[example_key][IMAGE_KEY][...],
                    seg=hf[example_key][SEGMENTATION_KEY][...],
                    dist=hf[example_key][DISTANCE_TRANSFORM_KEY][...],
                    dx=hf[example_key].attrs['dx']
                )

    def _in_dataset(self, example_key, dataset_key):
        return example_key in self.datasets[dataset_key]

    def in_training_dataset(self, example_key):
        """ Returns True if example key is in the training dataset
        """
        return self._in_dataset(example_key, TRAINING_DATASET_KEY)

    def in_validation_dataset(self, example_key):
        """ Returns True if example key is in the validation dataset
        """
        return self._in_dataset(example_key, VALIDATION_DATASET_KEY)

    def in_testing_dataset(self, example_key):
        """ Returns True if example key is in the testing dataset
        """
        return self._in_dataset(example_key, TESTING_DATASET_KEY)


class DatasetProxy:
    """ A proxy object attached to a level set machine learning model
    after fitting so that we can do things like `model.training_data[0]`"""

    def __init__(self, datasets_handler, dataset_key):

        self._datasets_handler = datasets_handler
        self._dataset_key = dataset_key

    def __len__(self):
        return len(self._datasets_handler.datasets[self._dataset_key])

    def __getitem__(self, item):

        if not isinstance(item, int):
            raise KeyError

        key = self._datasets_handler.datasets[self._dataset_key][item]
        return self._datasets_handler._get_example_by_key(key)

    def __iter__(self):
        return (self[i] for i in range(len(self)))
