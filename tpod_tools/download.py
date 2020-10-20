#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
"""Download datasets from one or more CVAT tasks.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time

import configargparse
import requests
from tqdm.auto import tqdm


def cvat_export_dataset(
    cvat_url,
    task_id,
    dataset_format,
    auth=None,
    position=0,
):
    """Download a dataset from CVAT"""

    with tqdm(
        desc="Exporting dataset for task {}".format(task_id),
        position=position,
        leave=False,
    ) as pbar:
        with requests.Session() as session:
            if auth:
                session.auth = auth

            url = "{}/api/v1/tasks/{}/dataset".format(cvat_url, task_id)
            params = {"format": dataset_format}
            creating = True

            try:
                # request export of the dataset and wait for it to be ready
                while True:
                    response = session.get(url, params=params, stream=True)
                    response.raise_for_status()

                    if response.status_code == 200:
                        # file is ready for download
                        break

                    if response.ok and creating:
                        params["action"] = "download"
                        creating = False
                        continue

                    time.sleep(1)
                    pbar.update(0)

                # switch progress bar from 'waiting for' to 'downloading' dataset
                pbar.reset()
                pbar.total = int(response.headers.get("Content-Length", 0))
                pbar.unit = "B"
                pbar.unit_scale = True
                pbar.unit_divisor = 1024
                pbar.set_description("Download dataset {}".format(task_id))

                # download exported dataset
                with open("dataset_{}.zip".format(task_id), "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        output_file.write(chunk)
                        pbar.update(len(chunk))

            except requests.exceptions.HTTPError as exc:
                tqdm.write("Failed exporting dataset {}: {}".format(task_id, exc))


def main():
    """Download dataset from CVAT (CLI)"""
    parser = configargparse.ArgParser(default_config_files=["~/.tpod-tools"])
    parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
    parser.add_argument("--url", required=True, help="base URL of CVAT installation")
    parser.add_argument("--username", help="CVAT login username")
    parser.add_argument("--password", help="CVAT login password")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument(
        "--format",
        default="pascal",
        help="dataset format [coco, pascal, labelme, mask, mot, tfrecord, yolo]",
    )
    parser.add_argument(
        "task_id", type=int, nargs="+", help="task id of dataset to download"
    )
    args = parser.parse_args()

    _format = {
        "coco": "COCO 1.0",
        "pascal": "PASCAL VOC 1.1",
        "labelme": "LabelMe 3.0",
        "mask": "Segmentation mask 1.1",
        "mot": "MOT 1.1",
        "tfrecord": "TFRecord 1.0",
        "yolo": "YOLO 1.1",
    }.get(args.format) or args.format

    _auth = (args.username, args.password) if args.username or args.password else None

    export_dataset = partial(
        cvat_export_dataset, args.url, dataset_format=_format, auth=_auth
    )

    with ThreadPoolExecutor() as p:
        for n, task in enumerate(args.task_id, 1):
            p.submit(export_dataset, task, position=n)


if __name__ == "__main__":
    main()
