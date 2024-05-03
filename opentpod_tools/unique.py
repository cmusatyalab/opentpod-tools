#!/usr/bin/env python3
#
#  Copyright (c) 2018-2024 Carnegie Mellon University
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
import random
from pathlib import Path

import datumaro as dm
import imagehash
from datumaro.components.dataset_base import IDataset
from datumaro.components.media import Image as dmImage
from datumaro.components.transformer import ItemTransform
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

    def __init__(
        self,
        extractor: IDataset,
        dedup_method: str = "sequential",
        threshold: int = DIFF_THRESHOLD,
        ratio: float = DEFAULT_RATIO,
        progress=None,
    ):
        super().__init__(extractor)
        self._method = dedup_method
        self._threshold = threshold
        self._ratio = ratio
        self._deduped = []
        self._progress = progress

    @staticmethod
    def _compute_image_hash(image):
        im = Image.fromarray(image.data)
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

        if self._progress is not None:
            self._progress.update()

        for check_hash in self.check_list:
            if (image_hash - check_hash) < self._threshold:
                return None

        self._deduped.append(image_hash)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=Path, default="unique", help="Output dataset"
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["sequential", "random", "exhaustive"],
        default="sequential",
        help="amount of effort spent to look for possible duplicates",
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

    datumaro_fixup(args.dataset)
    dataset = dm.Dataset.import_from(str(args.dataset))

    dataset_len = len(dataset)
    with tqdm(total=dataset_len) as pbar:
        dataset = dataset.transform(
            DedupTransform,
            dedup_method=args.method,
            threshold=args.threshold,
            ratio=args.ratio,
            progress=pbar,
        )

        # the transform is lazily executed when we look at the data items, in
        # this case calling len() actually triggers processing all transforms
        duplicates = dataset_len - len(dataset)
    print(f"Removed {duplicates} similar items")

    dataset.save(str(args.output), save_media=args.save_images)


if __name__ == "__main__":
    main()
