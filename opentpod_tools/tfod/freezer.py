#
#  Copyright (c) 2019-2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Freeze trained Tensorflow Object Detector model"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from logzero import logger


def freeze(model_dir, output_file_path):
    """Export TF model.
    Both the frozen graph and training artifacts are exported to allow
    inference and future training.

    Note: Since TF's object detection API is not using TF v2.0. We had to
    run the export script in a separate process with TF eager mode disabled.
    CVAT and object_detector.datasets enable TF eager mode for easy
    read/write TFrecord files, causing the following model export script to
    throw errors due to calls to tf.placeholder(). See more at:
    https://github.com/tensorflow/tensorflow/issues/18165

    When TF object detection has migrated to TF v2.0, something like train()
    can be done to directly call the export script as a python function.
    """
    model_dir = Path(model_dir)
    model_checkpoint_dir = _get_latest_model_ckpt_path(model_dir)
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = (
            "python"
            " -m opentpod_tools.tfod.wrappers.export_inference_graph"
            " --input_type=image_tensor"
            " --pipeline_config_path={}"
            " --trained_checkpoint_prefix={}"
            " --output_directory={}"
            " --alsologtostderr"
        ).format(
            model_dir / "pipeline.config",
            model_checkpoint_dir,
            temp_dir,
        )
        logger.info(
            """
===========================================

Exporting trained model with following command:

%s

===========================================
""",
            cmd,
        )
        process = subprocess.Popen(cmd.split())
        process.wait()

        # copy some useful training files to export as well
        shutil.copy2(model_dir / "label_map.pbtxt", temp_dir)

        output_file_path = Path(output_file_path)
        file_stem = os.fspath(output_file_path.parent / output_file_path.stem)
        logger.debug("Exporting to %s.zip", file_stem)
        shutil.make_archive(file_stem, "zip", temp_dir)


def _get_latest_model_ckpt_path(model_dir):
    candidates = [
        os.fspath(candidate) for candidate in model_dir.glob("**/model.ckpt*")
    ]
    max_step_model_path = candidates[0]
    max_steps = re.findall(r"model.ckpt-(\d+)", max_step_model_path)[0]
    for candidate_path in candidates:
        trained_steps = re.findall(r"model.ckpt-(\d+)", candidate_path)[0]
        if trained_steps > max_steps:
            max_step_model_path = candidate_path
    # the max_step_model_path now is a full of e.g.
    # .../model-ckpt-2000.data-00000-of-00001
    # however, for TF's export code, we need to give  .../model-ckpt-2000
    # as there are multiple files ending in .meta, .index, .data-...
    return os.path.splitext(max_step_model_path)[0]


def main():
    """CLI for freezing a model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Trained model")
    parser.add_argument(
        "-o", "--output", help="Frozen model name (defaults to 'model-dir.zip')"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.model_dir + ".zip"

    freeze(args.model_dir, args.output)


if __name__ == "__main__":
    main()
