#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: 2020-2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0
#
"""Filter datasets
"""

import argparse
from pathlib import Path

import datumaro as dm

from .utils import datumaro_fixup


def main():
    """Merge datasets with Datumaro"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Extra verbose logging"
    )
    parser.add_argument(
        "--filter-occluded",
        action="store_true",
        help="Do not drop hidden or occluded annotations",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Copy images instead of only references for filtered dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="filtered",
        help="Filtered dataset name (defaults to 'filtered')",
    )
    parser.add_argument(
        "dataset", type=Path, help="path of dataset to filter"
    )
    args = parser.parse_args()

    print("Importing", args.dataset)
    datumaro_fixup(args.dataset)
    dataset = dm.Dataset.import_from(str(args.dataset))

    if args.verbose:
        print("IMPORTED", dataset)

    if args.filter_occluded:
        print("- Removing occluded annotations")
        dataset = dataset.filter('/item/annotation[occluded="False"]', filter_annotations=True, remove_empty=True)

        if args.verbose:
            print("REMOVED OCCLUSIONS", dataset)

    # remove frames with no annotations
    print("- Removing empty frames")
    filtered_dataset = dataset.filter('//*', filter_annotations=True, remove_empty=True)

    if args.verbose:
        print("REMOVED EMPTY FRAMES", dataset)

    filtered_dataset.save(str(args.output), save_media=args.save_images)

if __name__ == "__main__":
    main()
