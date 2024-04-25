#!/usr/bin/env python3
#
#  Copyright (c) 2018-2020 Carnegie Mellon University
#  All rights reserved.
#
# Based on work by Junjue Wang.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Remove similar frames based on a perceptual hash metric
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import shutil
from pathlib import Path
from typing import Literal

import datumaro as dm
from datumaro.components.dataset_base import IDataset
from datumaro.components.media import Image as dmImage
from datumaro.components.transformer import ItemTransform

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import datumaro_fixup

DIFF_THRESHOLD = 10
DEFAULT_RATIO = 0.7


class DedupTransform(ItemTransform):
    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--method",
            dest="dedup_method",
            type=str,
            default="sequential",
            help="Comparison method [sequential, random, exhaustive]",
        )
        parser.add_argument(
            "--threshold",
            dest="threshold",
            type=int,
            default=DIFF_THRESHOLD,
            help="Similarity threshold",
        )
        parser.add_argument(
            "--ratio",
            dest="ratio",
            type=float,
            default=DEFAULT_RATIO,
            help="Ratio for random sample size to check against",
        )
        return parser

    def __init__(self, extractor: IDataset, dedup_method: str = "sequential", threshold: int = DIFF_THRESHOLD, ratio: float = DEFAULT_RATIO):
        super().__init__(extractor)
        self._method = dedup_method
        self._threshold = threshold
        self._ratio = ratio
        self._deduped = []

    @staticmethod
    def _compute_image_hash(image):
        print(image.path)
        im = Image.fromarray(image.data)
        #a = np.asarray(image)
        #im = Image.fromarray(a)
        return imagehash.phash(im)

    @property
    def check_list(self):
        if self._method == "sequential":
            return self._deduped[-1:]

        elif self._method == "exhaustive":
            return self._deduped

        else:  # self._method == "random":
            k = len(self._deduped) * self._ratio
            return random.sample(self._deduped, k)

    def transform_item(self, item):
        image_hash = self._compute_image_hash(item.media_as(dmImage))

        for check_hash in self.check_list:
            if (image_hash - check_hash) < self._threshold:
                return None

        self._deduped.append(image_hash)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default="unique", help="Output dataset")
    parser.add_argument(
        "-l",
        "--level",
        required=True,
        type=int,
        help="1(continuous checking) 2(random checking) 3(complete checking)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=DIFF_THRESHOLD,
        help="Threshold of difference",
    )
    parser.add_argument(
        "-r", "--ratio", type=float, default=DEFAULT_RATIO, help="Random ratio"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Copy images instead of only references for filtered dataset",
    )
    parser.add_argument("dataset", type=Path, help="Input dataset")

    args = parser.parse_args()

    if args.level == 1:
        method = "sequential"
    elif args.level == 2:
        method = "random"
    elif args.level == 3:
        method = "exhaustive"
    else:
        raise Exception("No suitable level found")


    datumaro_fixup(args.dataset)
    dataset = dm.Dataset.import_from(str(args.dataset))

    deduped_dataset = dataset.transform(
        DedupTransform,
        dedup_method=method,
        threshold=args.threshold,
        ratio=args.ratio,
    )

    duplicates = len(dataset) - len(deduped_dataset)
    print(f"Removed {duplicates} similar items")

    deduped_dataset.save(str(args.output), save_media=args.save_images)

    import sys
    sys.exit()

    annotations = json.loads(
        args.dataset.joinpath("annotations", "default.json").read_text()
    )

    base_image_list = []
    if args.level == 1:
        check_dup = functools.partial(checkDiff, base_image_list, args.threshold)

    elif args.level == 2:
        check_dup = functools.partial(checkDiffRandom, base_image_list, args.ratio, args.threshold)

    elif args.level == 3:
        check_dup = functools.partial(checkDiffComplete, base_image_list, args.threshold)

    result = []
    for i in tqdm(annotations["items"]):
        imgpath = i["image"]["path"]
        im = Image.open(imgpath)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)

        if base_image_list and check_dup(image_hash):
            continue

        base_image_list.append(image_hash)
        result.append(i)

    duplicates = len(dataset) - len(result)
    print(f"Removed {duplicates} similar items")

    data = annotations.copy()
    data["items"] = result

    #with args.output.joinpath("annotations", "default.json").open("w") as fp:
    #    json.dump(data, fp)


if __name__ == "__main__":
    main()
