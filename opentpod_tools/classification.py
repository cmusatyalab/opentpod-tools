#!/usr/bin/env python3
#
#  Copyright (c) 2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import os
import shutil

from PIL import Image


def prepare(resultpath, split):
    if os.path.exists(resultpath):
        shutil.rmtree(resultpath)
    os.mkdir(resultpath)
    if split:
        os.mkdir(os.path.join(resultpath, "train"))
        os.mkdir(os.path.join(resultpath, "val"))


def makedataset(resultpath, jsonpath):
    dic = json.loads(open(jsonpath).read())
    item2label = {}
    counter = 0
    for i in dic["categories"]["label"]["labels"]:
        dirname = i["name"]
        p = os.path.join(resultpath, dirname)
        dirpath = os.mkdir(p)
        item2label[counter] = p
        counter += 1

    uniqueid = 0
    for i in dic["items"]:
        imagepath = i["image"]["path"]
        path_piece = imagepath.split("/")[-1]
        store_name = str(uniqueid) + path_piece
        im = Image.open(imagepath)
        uniqueid += 1
        height = i["image"]["size"][0]
        width = i["image"]["size"][1]
        annoid = 0
        for j in i["annotations"]:
            lid = j["label_id"]
            name = str(annoid) + store_name
            annoid += 1
            storepath = os.path.join(item2label[lid], name)
            cropImg = im.crop(
                (
                    j["bbox"][0],
                    j["bbox"][1],
                    j["bbox"][0] + j["bbox"][2],
                    j["bbox"][1] + j["bbox"][3],
                )
            )
            cropImg.save(storepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", action="store_true", default=False)
    parser.add_argument(
        "-o", "--output", default="classification", help="Output dataset path"
    )
    parser.add_argument("-p", "--path", required=True, help="Input dataset path")
    args = parser.parse_args()
    prepare(args.output, args.split)
    if not args.split:
        jsonpath = os.path.join(args.path, "dataset", "annotations", "default.json")
        if not os.path.exists(jsonpath):
            msg = "Please check default.json path: " + str(jsonpath)
            raise Exception(msg)
        print(jsonpath)
        makedataset(args.output, jsonpath)
    else:
        trainjson = os.path.join(args.path, "dataset", "annotations", "train.json")
        if not os.path.exists(trainjson):
            msg = "Please check train.json path: " + str(trainjson)
            raise Exception(msg)
        valjson = os.path.join(args.path, "dataset", "annotations", "eval.json")
        if not os.path.exists(valjson):
            msg = "Please check eval.json path: " + str(valjson)
            raise Exception(msg)
        print(trainjson)
        print(valjson)
        makedataset(os.path.join(args.output, "train"), trainjson)
        makedataset(os.path.join(args.output, "val"), valjson)


if __name__ == "__main__":
    main()
