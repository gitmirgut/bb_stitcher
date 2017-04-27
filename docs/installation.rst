============
Installation
============
This part of the documentation covers the installation of the ``bb_stitcher``.

--------------------------------
(Recommended) Conda Installation
--------------------------------
The following steps will install all dependencies including opencv.

1. Download and install `Conda <https://conda.io/docs/install/quick.html>`_::

    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ conda update conda

2. Setup conda environment for ``bb_stitcher``::

    $ conda create --name bb_stitcher_env python=3
    $ source activate bb_stitcher_env
    (bb_stitcher_env)$ conda install --channel https://conda.binstar.org/menpo opencv3

3. Install the ``bb_stitcher``::

    (bb_stitcher_env) $ pip install git+https://github.com/BioroboticsLab/bb_stitcher.git

----------------
Pip Installation
----------------

.. important::
    The following requirements must be installed manually and cannot be installed by pip:

    * `OpenCV3 <https://github.com/opencv/opencv>`_
    * `opencv_contrib <https://github.com/opencv/opencv_contrib>`_

    Good Instruction for installing opencv with opencv_contrib package
    can be found under `pyimagesearch <http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/>`_.

Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^
Direct install from GitHub::

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


