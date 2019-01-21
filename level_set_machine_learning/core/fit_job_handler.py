import logging

from .datasets_handler import DatasetsHandler
from .temporary_data_handler import TemporaryDataHandler


def setup_logging():
    """ Sets up logging formatting, etc
    """
    from level_set_machine_learning import LevelSetMachineLearning
    logging.basicConfig(
        filename="{}-fit-log.txt".format(LevelSetMachineLearning.__name__),
        format="[%(asctime)s] [%(name)] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")


class FitJobHandler:
    """ Manages attributes and model fitting data/procedures
    """
    def __init__(self, model, data_file, imgs, segs, dx,
                 normalize_imgs_on_convert,
                 datasets_split, random_state, step, temp_data_dir,
                 regression_model_class, regression_model_kwargs):
        """ See :class:`level_set_machine_learning.LevelSetMachineLearning`
        for complete descriptions of parameters
        """
        # Set up logging formatting, file location, etc.
        setup_logging()

        # The LevelSetMachineLearning instance
        self.model = model

        # Initialize the iteration number
        self.iteration = 0

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
            h5_file=data_file, imgs=imgs, segs=segs, dx=dx,
            normalize_imgs_on_convert=normalize_imgs_on_convert)

        # Split the examples into corresponding datasets
        self.datasets_handler.assign_examples_to_datasets(
            training=datasets_split[0],
            validation=datasets_split[1],
            testing=datasets_split[2],
            random_state=random_state)

        # Initialize temp data handler for managing per-iteration level set
        # values, etc.
        self.temp_data_handler = TemporaryDataHandler(tmp_dir=temp_data_dir)
