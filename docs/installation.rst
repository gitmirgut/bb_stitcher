============
Installation
============
This part of the documentation covers the installation of the ``bb_stitcher``.

----------------
Pip Installation
----------------

(Special) Requirements
^^^^^^^^^^^^^^^^^^^^^^

The following requirements must be installed manually and cannot be installed by pip:

* `OpenCV3 <https://github.com/opencv/opencv>`_
* `opencv_contrib <https://github.com/opencv/opencv_contrib>`_

Good Instruction for installing opencv with opencv_contrib package
can be found under `pyimagesearch <http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/>`_.

Simple Installation
^^^^^^^^^^^^^^^^^^^
Direct install from github::

    pip install git+https://github.com/BioroboticsLab/bb_stitcher.git

Installation from source
^^^^^^^^^^^^^^^^^^^^^^^^
For developers the following setup instruction is recommended.

Get the source
""""""""""""""

You can clone the public repository::

    $ git clone https://github.com/BioroboticsLab/bb_stitcher.git

**Or**, download the tarball::

    $ wget curl -OL https://github.com/BioroboticsLab/bb_stitcher/tarball/master

Install the source
""""""""""""""""""
Enter the directory and install dependencies using::

    $ pip install -r requirements.txt

Then, install bb_stitcher using::

    $ pip install -e .


