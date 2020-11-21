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

import json
import os
import argparse
import random
import shutil

import imagehash
from PIL import Image
import numpy as np

from datumaro.components.project import Project

DIFF_THRESHOLD = 10
DEFAULT_RATIO = 0.7


def checkDiff(image_hash, base_image_hash, threshold):
    if base_image_hash is None:
        return True
    if image_hash - base_image_hash >= threshold:
        return True

    return False


def checkDiffComplete(image_hash, base_image_list, threshold):
    if len(base_image_list) <= 0:
        return True
    for i in base_image_list:
        if not checkDiff(image_hash, i, threshold):
            return False
    return True


def checkDiffRandom(image_hash, base_image_list, check_ratio, threshold):
    if len(base_image_list) <= 0:
        return True
    check_length = int(len(base_image_list) * check_ratio)
    new_list = []
    new_list.extend(range(len(base_image_list)))
    random.shuffle(new_list)
    for i in new_list[:check_length]:
        if not checkDiff(image_hash, base_image_list[i], threshold):
            return False
    return True


def contProcess(dic, threshold):
    base_image_hash = None
    nodup = []
    print(len(dic["items"]))
    for i in dic["items"]:
        imgpath = i["image"]["path"]
        im = Image.open(imgpath)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiff(image_hash, base_image_hash, threshold):
            base_image_hash = image_hash
            nodup.append(i)

    return nodup


def completeProcess(dic, threshold):
    base_image_list = []
    nodup2 = []
    print(len(dic["items"]))
    for i in dic["items"]:
        imgpath = i["image"]["path"]
        im = Image.open(imgpath)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiffComplete(image_hash, base_image_list, threshold):
            base_image_list.append(image_hash)
            nodup2.append(i)
    return nodup2


def randomProcess(dic, ratio, threshold):
    if ratio < 0 or ratio > 1:
        raise Exception("Random ratio should between 0 and 1")
    base_image_list2 = []
    nodup3 = []
    print(len(dic["items"]))
    for i in dic["items"]:
        imgpath = i["image"]["path"]
        im = Image.open(imgpath)
        a = np.asarray(im)
        im = Image.fromarray(a)
        image_hash = imagehash.phash(im)
        if checkDiffRandom(image_hash, base_image_list2, ratio, threshold):
            base_image_list2.append(image_hash)
            nodup3.append(i)
    return nodup3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--level",
        required=True,
        type=int,
        help="1(continuous checking) 2(random checking) 3(complete checking)",
    )
    parser.add_argument("-p", "--path", required=True, help="Input dataset path")
    parser.add_argument("-o", "--output", default="unique", help="Output dataset path")
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

    args = parser.parse_args()

    if args.level > 3 or args.level < 1:
        raise Exception("No suitable level found")
    ANNOPATH = os.path.join(args.path, "dataset", "annotations")
    FRAMEJSON = os.path.join(ANNOPATH, "default.json")

    dic = json.loads(open(FRAMEJSON).read())

    if args.level == 1:
        result = contProcess(dic, args.threshold)
        print(len(result))
        data = {}
        data["categories"] = dic["categories"]
        data["items"] = result
        data["info"] = dic["info"]
    elif args.level == 2:
        result = randomProcess(dic, args.ratio, args.threshold)
        print(len(result))
        data = {}
        data["categories"] = dic["categories"]
        data["items"] = result
        data["info"] = dic["info"]
    elif args.level == 3:
        result = completeProcess(dic, args.threshold)
        print(len(result))
        data = {}
        data["categories"] = dic["categories"]
        data["items"] = result
        data["info"] = dic["info"]

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    p = Project.generate(
        args.output,
        {
            "project_name": args.output,
        },
    )
    p.make_dataset().save()

    RESULTPATH = os.path.join(args.output, "dataset", "annotations")
    path = os.path.join(RESULTPATH, "default.json")
    fp = open(path, "w")
    json.dump(data, fp)
    fp.close()


if __name__ == "__main__":
    main()
