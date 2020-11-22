#!/usr/bin/env python3
#
#  Copyright (c) 2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Export dataset for Google AutoML training
"""

import argparse
import json
import os
import shutil


def prepareOutput(outputpath="result"):
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)
    os.mkdir(outputpath)
    datapath = os.path.join(outputpath, "data")
    os.mkdir(datapath)


def obtainData(jsonpath, outputpath, bucketpath):
    dic = json.loads(open(jsonpath).read())
    item2label = {}
    counter = 0
    for i in dic["categories"]["label"]["labels"]:
        item2label[counter] = i["name"]
        counter += 1

    uniqueid = 0
    info = []
    for i in dic["items"]:
        imagepath = i["image"]["path"]
        path_piece = imagepath.split("/")[-1]
        store_name = str(uniqueid) + path_piece
        uniqueid += 1
        height = i["image"]["size"][0]
        width = i["image"]["size"][1]
        dst = os.path.join(outputpath, "data", store_name)
        shutil.copyfile(imagepath, dst)
        prefix = "UNASSIGNED,{}".format(os.path.join(bucketpath, "data", store_name))
        for j in i["annotations"]:
            lid = j["label_id"]
            itemname = item2label[lid]
            xmin = float(j["bbox"][0]) / width
            ymin = float(j["bbox"][1]) / height
            xmax = float(j["bbox"][0] + j["bbox"][2]) / width
            ymax = float(j["bbox"][1] + j["bbox"][3]) / height
            info.append(f"{prefix},{itemname},{xmin},{ymin},,,{xmax},{ymax},,")

    outputcsv = open(os.path.join(outputpath, "info.csv"), "w+")
    outputcsv.write("\n".join(info))
    outputcsv.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        required=True,
        help="Google cloud bucket name (please include gs://)",
    )
    parser.add_argument("-p", "--path", required=True, help="Input dataset path")
    args = parser.parse_args()
    prepareOutput()
    jsonpath = os.path.join(args.path, "dataset", "annotations", "default.json")
    obtainData(jsonpath, "result", args.bucket)


if __name__ == "__main__":
    main()
