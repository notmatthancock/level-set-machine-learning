Setup and Dependencies
----------------------

These tools use mostly Python with a peppering of C/C++ and maybe a bit of
Fortran.

Python Libraries
~~~~~~~~~~~~~~~~

The main tool is `pylidc <https://pylidc.github.io>`_. By installing 
`pylidc` (follow the instructions at the link), most of the Python library 
dependencies (e.g., NumPy, SciPy, etc.) will be installed automatically.

If this doesn't work, you might have to install 
each Python dependency by manual specification::

    pip install sqlalchemy numpy scipy matplotlib pydicom scikit-image h5py

And once these are installed, then install `pylidc`::

    pip install pylidc

On the other hand, if you want to have a development version of `pylidc` so 
that you can pull the most recent changes or contribute changes, just 
clone the repository::

    git clone https://github.com/pylidc/pylidc.git

Make sure you add the path where you cloned `pylidc` to your `PYTHONPATH`
environment variable in your `.bashrc` or `.bash_profile` file.


LIDC DICOM Data
~~~~~~~~~~~~~~~

Next, assuming you plan to work the LIDC image data, you should download all
of the DICOM data, which `can be found here <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_
(at least, as of 4/17/2018). You should probably download this data to some 
external hard disk, since the data totals about 124 GB. It takes some time to
download it all, so you might consider letting it download overnight.

Note that this is the *raw* DICOM data. The `pylidc` library has utilities for
converting this data to more usable NumPy array data for image-processing.

Once you've downloaded this data, you should follow the configuration setup 
so that `pylidc` can use the DICOM data. `Instructions for that are here <https://pylidc.github.io/install.html#dicom-file-directory-configuration>`_

Once successful, you should consider fooling around with the tutorials in the
`pylidc` documentation to make sure everything is working and to get a feel 
for `pylidc`.
