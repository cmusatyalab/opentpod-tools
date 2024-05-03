<!--
SPDX-FileCopyrightText: 2020 Carnegie Mellon University

SPDX-License-Identifier: Apache-2.0
-->

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
opentpod-tools in that virtualenv.  You can start up a shell using the
installed virtualenv environment with `poetry shell` and work from there.


## Usage

The following assume that `opentpod-tools` has been installed globally, or you
are running it from within a virtualenv (see `poetry run`/`poetry shell`).

Download, merge and cleanup datasets.

```sh
    # upload videos to CVAT, and label them

    # download labeled datasets
    $ tpod-download [--project|--task|--job] <dataset0> .. <datasetN>

    # merge datasets
    $ datum merge [-o merged] <dataset0> .. <datasetN>

    # filter frames with no annotations (and optionally annotated occlusions)
    $ tpod-filter [--filter-occluded] [-o filtered] merged

    # remove duplication
    $ tpod-unique [-m sequential|random|complete] [-o unique] merged [-t 10 -r 0.7]
    # -m --method: sequential, only check against the last 'unique' image
    #              random, check against random subset of unique image list with [-r/--ratio]
    #              exhaustive, check each new image against all chosen unique images
    # -t --threshold: the difference between current image and unique image(s), default = 10

    # split into training and validation subsets
    $ datum transform -t random_split -o split unique -- -s train:0.9 -s val:0.1 [-s test:...]
    $ rmdir split/images || true
```

Explore the dataset.

```sh
    # high level information (# image in trainingset and evaluation set)
    $ datum dinfo split

    # detailed statistics (distribution of labels, area of labeled features, etc.)
    $ datum stats split
```

Train a yolo object detector.

```sh
    # export to yolo_ultralytics format
    $ datum convert -i split -f yolo_ultralytics -o yolo-dataset -- --save-media

    # install Ultralytics YOLOv8 trainer (may already be installed?)
    $ pip install ultralytics

    # train an object detector model
    $ yolo detect train data=$(pwd)/yolo-dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 project=yolo-project
```


<!---
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
-->
