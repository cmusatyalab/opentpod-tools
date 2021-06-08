# SPDX-FileCopyrightText: 2020 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil

import torch
from torchvision import models

from .pytorch_mobilenet import prepareData as mobilenetprepare
from .pytorch_resnet import prepareData as resnetprepare


def writeInfo(savedir, class_names):
    infopath = os.path.join(savedir, "info.txt")
    fp = open(infopath, "w+")
    for i in range(len(class_names)):
        fp.write(str(i) + ":" + class_names[i] + "\n")
    fp.close()


def mobilenet(data_dir, output_path, epoch):
    model_ft = models.mobilenet_v2(pretrained=True)
    model, class_names = mobilenetprepare(data_dir, epoch, model_ft)
    torch.save(model, os.path.join(output_path, "result.pth"))
    writeInfo(output_path, class_names)
    return


def resnet50(data_dir, output_path, epoch):
    model_ft = models.resnet50(pretrained=True)
    model, class_names = resnetprepare(data_dir, epoch, model_ft)
    torch.save(model, os.path.join(output_path, "result.pth"))
    writeInfo(output_path, class_names)
    return


def resnet18(data_dir, output_path, epoch):
    model_ft = models.resnet18(pretrained=True)
    model, class_names = resnetprepare(data_dir, epoch, model_ft)
    torch.save(model, os.path.join(output_path, "result.pth"))
    writeInfo(output_path, class_names)
    return


def main():
    model2func = {"mobilenet": mobilenet, "resnet50": resnet50, "resnet18": resnet18}
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Input dataset path")
    parser.add_argument("-m", "--model", default="mobilenet")
    parser.add_argument("-o", "--output", required=True, help="Output dataset path")
    parser.add_argument("-e", "--epoch", type=int, default=25)
    # parser.add_argument('-l', '--log', default='logger')
    args = parser.parse_args()
    data_dir = args.path
    output_path = args.output
    print(data_dir)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    model2func[args.model.lower()](data_dir, output_path, args.epoch)


if __name__ == "__main__":
    main()
