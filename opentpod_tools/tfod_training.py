#!/usr/bin/env python3
#
#  Copyright (c) 2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Tensorflow Object Detection

Download models and train a detector.
"""

import argparse

from opentpod_tools.tfod import REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=REGISTRY.keys(), required=True)
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="TFrecord formatted 'train' and 'eval' sets",
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Trained model")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fine-tune-checkpoint", help="Pretrained model to start from")
    args = parser.parse_args()

    # collect arguments, drop unspecified ones
    config = dict((k, v) for k, v in vars(args).items() if v is not None)

    instance = REGISTRY[args.model](config)
    instance.prepare()
    instance.train()


if __name__ == "__main__":
    main()
