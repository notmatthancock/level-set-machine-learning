import contextlib
import os
import time

import h5py


# The name of the temporary hdf5 file to be created
TMP_H5_FILE_NAME = 'tmp.h5'

# The name of the lock file file used to indicate locked tmp h5
TMP_H5_LOCK_FILE_NAME = '{}.lock'.format(TMP_H5_FILE_NAME)

# The name of sub-directory where temporary data are stored
TMP_DIR_NAME = 'tmp'


class TemporaryDataHandler:
    """ Handles internal management of temporary data created during
    the fitting process of a LevelSetMachineLearning model
    """
    def __init__(self, tmp_dir=os.path.curdir):
        """ Initialize a model fitting data manager

        Parameters
        ----------
        tmp_dir: str, default=os.curdir
            Temporary data will be managed in a temporary folder under
            this directory

        """
        tmp_data_location = os.path.join(tmp_dir, TMP_DIR_NAME)
        self.tmp_data_location = os.path.abspath(tmp_data_location)

    def make_tmp_location(self):
        """ Make the folder that will hold the temporary data for model fitting
        """
        # Raises FileExistsError if it already exists
        os.makedirs(self.tmp_data_location)

    def remove_tmp_data(self):
        """ Delete the temporary data folder and contents
        """
        os.removedirs(self.tmp_data_location)

    def _get_h5_file_path(self):
        """ Returns the full path to the hdf5 file holding the temporary data
        """
        return os.path.join(self.tmp_data_location, TMP_H5_FILE_NAME)

    @contextlib.contextmanager
    def open_h5_file(self, lock=False):
        """ Opens the hdf5 file housing the temporary data in the form of
        a context manager that handles locking, closing, etc.

        Parameters
        ----------
        lock: bool, default=False
            If True, then a lock file is created that indicates to other
            processes to not open the file. This is useful for do writes.

        Example
        -------
        with tmp_mgmt.open_h5_file(lock=True) as h5_file:
            img = h5_file['training-data-example/img'][...]
            # etc.

        """

        # Check for the lock and wait for release if necessary
        self._wait_for_h5_file_unlock()

        try:
            if lock:
                self._create_h5_file_lock()
            h5 = h5py.File(self._get_h5_file_path())
            yield h5
        finally:
            h5.close()
            if lock:
                # Remove the lock if necessary
                self._remove_h5_file_lock()

    def _wait_for_h5_file_unlock(self):
        """ Loop and sleep until the lock file is released
        """
        while not self._can_open_h5_file():
            time.sleep(0.1)

    def _get_lock_file_path(self):
        """ Returns the full path to the lock file
        """
        return os.path.join(self.tmp_data_location, TMP_H5_LOCK_FILE_NAME)

    def _can_open_h5_file(self):
        """ Returns True if the lock file does not exist
        """
        lock_file = self._get_lock_file_path()
        return not os.path.exists(lock_file)

    def _create_h5_file_lock(self):
        """ Creates an empty lock file
        """
        lock_file = self._get_lock_file_path()

        # Create an appropriately named empty file
        with open(lock_file, 'w'):
            pass

    def _remove_h5_file_lock(self):
        """ Remove the lock file and unlock the temp data file
        """
        lock_file = self._get_lock_file_path()
        os.remove(lock_file)
