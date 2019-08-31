

class ModelNotFit(Exception):
    """ Raised when trying access properties or methods that require a fitted
    model
    """


class ModelAlreadyFit(Exception):
    """ Raised when attempting a model that has already been fitted
    """
