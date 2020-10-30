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
    poetry install

This will create a virtualenv with all the dependencies and installs
opentpod-tools in that virtualenv.  You can start up a shell with the right
virtualenv environment with ``poetry shell`` and work from there.
