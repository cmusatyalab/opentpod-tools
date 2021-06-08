# SPDX-CopyrightText: 2017 PyTorch
# SPDX-CopyrightText: 2020 Carnegie Mellon University
#
# SPDX-License-Identifier: BSD-3-Clause

# License: BSD
# Author: Sasank Chilamkurthy
# © Copyright 2017, PyTorch
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
#
# Modified: Zhen Luan zluan@andrew.cmu.edu

import argparse
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms


def getItemInfo(infopath):
    infodict = {}
    fp = open(infopath, "r")
    info = fp.read().split()
    for i in info:
        index2item = i.split(":")
        # print(index2item)
        infodict[int(index2item[0])] = index2item[1]
    fp.close()
    return infodict


def classified(image_path, modeldir):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    modelpath = os.path.join(modeldir, "result.pth")
    infopath = os.path.join(modeldir, "info.txt")
    infodict = getItemInfo(infopath)
    model_ft = torch.load(modelpath)
    input_image = Image.open(image_path)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model_ft.to("cuda")

    with torch.no_grad():
        output = model_ft(input_batch).numpy()[0]

    resultIndex = np.argmax(output)
    return infodict[resultIndex]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Check image path")
    parser.add_argument("-p", "--path", required=True, help="Model path")
    args = parser.parse_args()
    item = classified(args.image, args.path)
    print(item)


if __name__ == "__main__":
    main()
