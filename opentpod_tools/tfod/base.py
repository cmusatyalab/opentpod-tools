#
#  Copyright (c) 2019-2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Tensorflow Object Detection API provider"""

import os
import pathlib
import re
import shutil
from string import Template

from logzero import logger

from . import utils


class TFODDetector:
    """Tensorflow Object Detection API
    See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md
    """  # noqa pylint: disable=line-too-long

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model_name = cls.__name__.lower()
        if not model_name.startswith("tfod"):
            utils.REGISTRY[model_name] = cls

    TRAINING_PARAMETERS = {}

    def __init__(self, config):
        """Expected directory layout
            +train-data
                -label_map file
                -train TFRecord file
                -eval TFRecord file
            +models
                + model
                    -pipeline config file
                    +train
                    +eval
        Arguments:
            config: contains input_dir, output_dir, and training parameters
        Output:
            ``${output_dir}/status`` file: training status
            ``${output_dir}/train.log`` file: training log file
        """
        super().__init__()
        self._config = config
        self._input_dir = pathlib.Path(self._config["input_dir"])
        self._output_dir = pathlib.Path(self._config["output_dir"])

        # find appropriate model to finetune from
        self.cache_pretrained_model()

        # fill in on-disk file structure to config
        self._config["pipeline_config_path"] = os.fspath(
            self._output_dir / "pipeline.config"
        )
        self._config["train_input_path"] = os.fspath(self._input_dir / "train.tfrecord")
        self._config["eval_input_path"] = os.fspath(self._input_dir / "eval.tfrecord")
        self._config["label_map_path"] = os.fspath(self._input_dir / "label_map.pbtxt")

    @property
    def training_parameters(self):
        return self.TRAINING_PARAMETERS

    @property
    def pretrained_model_cache_entry(self):
        return self.__class__.__name__ + "-pretrained-model"

    @property
    def pretrained_model_url(self):
        raise NotImplementedError()

    @property
    def pipeline_config_template(self):
        raise NotImplementedError()

    def cache_pretrained_model(self):
        """Download and cache pretrained model if not existed."""
        if utils.get_cache_entry(self.pretrained_model_cache_entry) is None:
            logger.info(
                "downloading and caching pretrained model from tensorflow website"
            )
            logger.info("url: %s", self.pretrained_model_url)
            utils.download_and_extract_url_tarball_to_cache_dir(
                self.pretrained_model_url, self.pretrained_model_cache_entry
            )

    def get_pretrained_model_checkpoint(self):
        cache_entry_dir = utils.get_cache_entry(self.pretrained_model_cache_entry)
        potential_pretained_model_files = list(cache_entry_dir.glob("**/model.ckpt*"))
        if len(potential_pretained_model_files) == 0:
            raise ValueError(f"Failed to find pretrained model in {cache_entry_dir}")
        fine_tune_model_dir = potential_pretained_model_files[0].parent
        return os.fspath(fine_tune_model_dir / "model.ckpt")

    def prepare_config(self):
        # num_classes are the number of classes to learn
        with open(self._config["label_map_path"]) as f:
            content = f.read()
            labels = re.findall(r"\tname: '(.+?)'\n", content)
            self._config["num_classes"] = len(labels)

        # fine_tune_checkpoint should point to the path of the checkpoint from
        # which transfer learning is done
        if self._config.get("fine_tune_checkpoint", None) is None:
            self._config["fine_tune_checkpoint"] = (
                self.get_pretrained_model_checkpoint()
            )

        # use default values for training parameters if not given
        for parameter, value in self.training_parameters.items():
            if parameter not in self._config:
                self._config[parameter] = value

    def prepare_config_pipeline_file(self):
        pipeline_config = Template(self.pipeline_config_template).substitute(
            **self._config
        )
        with open(self._config["pipeline_config_path"], "w") as f:
            f.write(pipeline_config)

    def prepare(self):
        """Prepare files needed for training."""
        self.prepare_config()

        # copy label map to trained model dir
        os.makedirs(self._config["output_dir"])
        shutil.copy2(self._config["label_map_path"], self._config["output_dir"])

        # create pipeline config in trained model dir
        self.prepare_config_pipeline_file()

    def _check_training_data_dir(self, FLAGS):
        """Check training directory's data is valid

        Fail if missing files, or # of training/eval examples used is not positive
        """
        training_data_dir = self._input_dir
        assert (training_data_dir / "label_map.pbtxt").exists()
        assert (training_data_dir / "label_map.pbtxt").stat().st_size > 0
        assert (training_data_dir / "train.tfrecord").exists()
        assert (training_data_dir / "eval.tfrecord").exists()

    def train(self):
        """Launch training using tensorflow object detection API."""
        from absl import flags
        from object_detection import model_main as continuous_train_and_eval_model

        # TF uses absl to get command line flags
        FLAGS = flags.FLAGS
        # argv[0] is treated as program name, therefore not parsed
        argv = [
            "",
            "--pipeline_config_path={}".format(self._config["pipeline_config_path"]),
            f"--model_dir={self._output_dir}",
            "--alsologtostderr",
        ]
        logger.info(
            """
===========================================

launching training with the following parameters:

    %s

===========================================
""",
            "\n    ".join(argv),
        )
        FLAGS(argv)
        self._check_training_data_dir(FLAGS)
        continuous_train_and_eval_model.main([])
