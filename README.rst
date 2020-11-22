OpenTPOD tools
==========

Collection of command line tools to assist with extracting, merging, and
training datasets from a CVAT installation.

The basis of this code originates from the Tool for Painless Object Detection
(OpenTPOD) developed by Junjue Wang.

We also pull in `datumaro <https://github.com/openvinotoolkit/datumaro>`_ which
is the backend used by CVAT to handle reading, writing, conversion, and merging
for various dataset formats.


Configuration file
------------------

You can create a config file in your home directory named ``.opentpod-tools`` with
common settings such as the CVAT installation base url, username, and password.

.. code-block:: cfg

    [cvat]
    url = http://localhost:8080
    username = user
    password = pass


Building
--------

This is my first attempt at using `Poetry <https://python-poetry.org>`_ to manage
python package dependencies, so I may be doing everything wrong.

But it should be possible to locally build this package as follows,

.. code-block:: sh

    # install poetry, see https://python-poetry.org/docs/
    # Make sure you install for python3
    #
    # I used (the not recommended way): pip3 install --user poetry

    git clone https://github.com/cmusatyalab/opentpod-tools.git
    cd opentpod-tools

    # run either of the following
    poetry install              # this installs tensorflow dependencies
    poetry install -E tf-gpu    # this will install tensorflow-gpu

This will create a virtualenv with all the dependencies and installs
opentpod-tools in that virtualenv.  You can start up a shell with the right
virtualenv environment with ``poetry shell`` and work from there.

Whenever you update your checked out source tree, it is useful to re-run
poetry install to pull in any updated dependencies as described in the new
poetry.lock file.

.. code-block:: sh

    cd opentpod-tools
    git pull
    poetry install --remove-untracked [-E tf-gpu]


Usage
-----

.. code-block:: sh

    # upload videos to CVAT, and label

    # download labeled datasets
    tpod-download <dataset0> .. <datasetN>

    # merge datasets
    tpod-merge -o merged <dataset0> .. <datasetN>

    # remove duplication
    tpod-unique -l 1 -o unique -p merged [-t 10 -r 0.7]
    -l --level: 1 continuous checking, always check the last unique image
                2 random checking, generate random set of unique image list with [-r/--ratio]
                3 complete checking, check the complete unique image list
    -t --threshold: the difference between current image and unique image(s), default = 10

    # split into training and testing subsets
    datum project transform -p unique -o split -t random_split -- -s train:0.9 -s eval:0.1

    # export to tfrecord format
    datum project export -p split -f tf_detection_api -o tfrecord

    # export to dataset for pytorch classification
    tpod-class [-s] -p split -o classification
    -s --split: the flag used to check whether the input directory has been splitted into training and testing subsets

    # export to dataset for google auto ml object detection (not completely done yet)
    tpod-google-automl-od -b <busket name on google cloud platform> -p unique

