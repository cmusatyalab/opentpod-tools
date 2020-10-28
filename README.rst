TPOD tools
==========

Collection of command line tools to assist with extracting, merging, and
training datasets from a CVAT installation.

The basis of this code originates from the Tool for Painless Object Detection
(OpenTPOD) developed by Junjue Wang.

We also pull in [datumaro][https://github.com/openvinotoolkit/datumaro] which
is the backend used by CVAT to handle reading, writing, conversion, and merging
for various dataset formats.


configuration file
------------------

You can create a config file in your home directory named `.opentpod-tools` with
common settings such as the CVAT installation base url, username, and password.

    [cvat]
    url = http://localhost:8080
    username = user
    password = pass
