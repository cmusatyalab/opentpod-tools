# OpenTPOD tools

Collection of command line tools to assist with extracting, merging, and
training datasets from a CVAT installation.

The basis of this code originates from the Tool for Painless Object Detection
(OpenTPOD) developed by Junjue Wang.

We also pull in [datumaro](https://github.com/openvinotoolkit/datumaro) which
is the backend used by CVAT which handles reading, writing, conversion, and
merging of various dataset formats.


## Configuration file

You can create a config file in your home directory named `.opentpod-tools` with
common settings such as the CVAT installation base url, username, and password.

```cfg
    [cvat]
    url = http://localhost:8080
    username = user
    password = pass
```


## Installation

```sh
    # set up a virtualenv with a newer pip
    $ python3 -m venv venv
    $ venv/bin/pip install --upgrade pip
    $ venv/bin/pip install git+https://github.com/cmusatyalab/opentpod-tools.git
```


## Building from source

This is my first attempt at using [Poetry](https://python-poetry.org) to manage
python package dependencies, so I may be doing everything wrong.

It should be possible to locally build this package as follows,

```sh
    # install poetry, see https://python-poetry.org/docs/
    # Make sure you install for python3
    #
    # I used (the not recommended way): pip3 install --user poetry
    $ git clone https://github.com/cmusatyalab/opentpod-tools.git
    $ cd opentpod-tools
    $ poetry install
```

This will create a virtualenv with all the dependencies and installs
opentpod-tools in that virtualenv.  You can start up a shell with the right
virtualenv environment with `poetry shell` and work from there.

Whenever you update your checked out source tree, it is useful to re-run
poetry install to pull in any updated dependencies as described in the new
`poetry.lock` file. If there is a merge conflict on the `poetry.lock` file
you can remove it and re-run `poetry install` to create a new conflict-free
version.

```sh
    cd opentpod-tools
    git pull
    poetry install
```

Note: Ubuntu 20.04 only has python3.8 by default which we currently don't
support because some of our dependencies don't support it yet. Tensorflow
(1.85) is not installable with python-3.8 and torch/torchvision have a
dependency (dataclasses) that locks us at python-3.6.

As a workaround you can install the python-3.6 release from the deadsnakes PPA.

```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6 python3.6-dev
```


## Usage

The following assume that `opentpod-tools` has been installed globally, or you
are running it from within a virtualenv (see `poetry run`/`poetry shell`).

Download, merge and cleanup datasets.

```sh
    # upload videos to CVAT, and label them

    # download labeled datasets
    $ tpod-download <dataset0> .. <datasetN>

    # merge datasets
    $ tpod-merge -o merged <dataset0> .. <datasetN>

    # remove duplication
    $ tpod-unique -l 1 -o unique -p merged [-t 10 -r 0.7]
    # -l --level: 1 continuous checking, always check the last unique image
    #             2 random checking, generate random set of unique image list with [-r/--ratio]
    #             3 complete checking, check the complete unique image list
    # -t --threshold: the difference between current image and unique image(s), default = 10

    # split into training and testing subsets
    $ datum project transform -p unique -o split -t random_split -- -s train:0.9 -s eval:0.1
```

Explore the dataset.

```sh
    # high level information (# image in trainingset and evaluation set)
    $ datum project info -p split

    # detailed statistics (distribution of labels, area of labeled features, etc.)
    $ datum project stats -p split
```

Train a tensorflow object detector.

```sh
    # export to tfrecord format
    $ datum project export -p split -f tf_detection_api -o tfrecord -- --save-images

    # train model and optionally freeze as 'new_model.zip'
    $ tpod-tfod-training --model faster_rcnn_resnet101 --input-dir tfrecord --output-dir new_model [--freeze]

    # visualize progress with tensorboard (default port is 6006)
    $ tensorboard --logdir=new_model --host=localhost --port=default

    # freeze model if not already frozen after training
    $ tpod-tfod-freeze --model-dir new_model --output new_model.zip
```

Train Pytorch classification model.

```sh
    # export to dataset for pytorch classification
    $ tpod-class [-s] -p split -o classification
    # -s --split: the flag used to check whether the input directory has been
    #    splitted into training and testing subsets

    # train pytorch classification model (NOTE: please split the datasets to
    # train and val first, and use tpod-class -s to obtain the required dataset)
    $ tpod-pytorch-class -p classification -o model [-m <model name>] [-e <echop number>]
    # -m --model: pytorch classification model name
    #     options: mobilenet, resnet50, resnet18 (not case sensitive), default = mobilenet
    # -e --epoch: default = 25

    # obtain classified result with input image
    $ tpod-pytorch-class-test -i <image path> -p model
```

Export for Google AutoML object detection training.

```sh
    # export to dataset for google auto ml object detection (not completely done yet)
    $ tpod-google-automl-od -b <bucket name on google cloud platform> -p unique
```
