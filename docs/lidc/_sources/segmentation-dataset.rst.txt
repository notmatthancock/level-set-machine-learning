Making a dataset for segmentation
=================================

Here we'll show the steps for transforming the DICOM data in the LIDC
dataset into subvolumes of bounding boxes about the nodules, using the 
`pylidc` library.

* `Step 1: collecting nodules with four annotations`_
* `Step 2: combining annotations and storing the data`_

Step 1: collecting nodules with four annotations
------------------------------------------------

First we'll collect all nodules for which there are four annotations. We'll 
create a list where each element of the list is itself a list of length 4,
corresponding to the Annotation ids of a nodule.

.. literalinclude:: /_static/code/get_nodules_with_four_annotations.py
    :language: python

Source: `get_nodules_with_four_annotations.py <_static/code/get_nodules_with_four_annotations.py>`_

You might see messages indicating failure to group nodules into groups <= 4 
in some cases. These might be tossed out, or manually inspected. These cases 
can occur when, for example, one radiologist annotates a nodule as two 
distinct nodules, while the others do not.

In any case, we've saved the ids into a matrix stored in NumPy format in the 
file called `annotation_ids_group_size_4.npy`. This matrix is shaped 
897-ish by 4. Each row is 4 `Annotation` ids for pylidc.

Let's load it back in::
    
    ids = np.load('annotations_ids_group_size_4.npy')
    anns = pl.query(pl.Annotation).filter(pl.Annotation.id.in_( ids[0] )).all()

We've fetched the first four annotations that refer to a single a nodule, 
which are stored in a list called `anns`. Each element of `anns` is a 
`pylidc.Annotation` object, with all the bells and whistles that come with it,
for example::

    for a in anns:
        print(a.scan.patient_id, a.malignancy)

This will print off the same patient id since the annotations all belong to 
the same scan, but print 4 different malignancy values corresponding to each 
of the 4 annotations of the nodule.

Step 2: combining annotations and storing the data
--------------------------------------------------

We will use the `pylidc consensus <https://pylidc.github.io/tuts/consensus.html>`_ 
function to combine multiple annotations to a single boolean volume. The 
consensus function accepts a list of :code:`Annotation` objects and 
returns (1) the bounding box of the common reference frame, (2) the consensus 
boolean-valued volume consolidation of the annotations, and (3) optionally, a 
list of boolean-valued volumes of the individual annotations. The bounding 
box returned will be used to index into the image volume at the correct 
location. We will store the data in `hdf5 format 
using h5py <https://www.h5py.org/>`_.

This accomplished in the following Python script:

.. literalinclude:: /_static/code/create_seg_data.py
    :language: python

Source: `create_seg_data.py <_static/code/create_seg_data.py>`_

The resulting file is only about 800 MB, compared to the 124 GB of the full 
LIDC dataset of all the slices, since we've sampled only sub-volumes from 
full CT image volumes and since we've used compression with h5py. Note 
that we store along with each image and segmentation volume, a number of 
meta attributes, such as the pixel and slice spacings. These are stored 
in the :code:`attr` dictionary of each example.

Note that the previous code might take some time to run 
and create all the volumes, and an overnight run might be a good idea.

Let's see how we might load and use the data after it is created:
    
.. literalinclude:: /_static/code/seg_data_demo.py
    :language: python

Source: `seg_data_demo.py <_static/code/seg_data_demo.py>`_
