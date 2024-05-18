#!/usr/bin/env python3
#
#  Copyright (c) 2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Download datasets from one or more CVAT tasks.
"""

import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import urljoin

import configargparse
import requests
from requests.exceptions import RequestException
from tqdm.auto import tqdm


def cvat_export_dataset(
    cvat_params, id_, output, dataset_format, class_="task", progress=None
):
    """Download a dataset from CVAT"""

    _format = {
        "datumaro": "Datumaro 1.0",
        "coco": "COCO 1.0",
        "pascal": "PASCAL VOC 1.1",
        "labelme": "LabelMe 3.0",
        "mask": "Segmentation mask 1.1",
        "mot": "MOT 1.1",
        "tfrecord": "TFRecord 1.0",
        "yolo": "YOLO 1.1",
    }.get(dataset_format, dataset_format)

    with requests.Session() as session:
        cvat_url, auth = cvat_params
        session.auth = auth

        url = urljoin(cvat_url, f"api/{class_}s/{id_}/dataset")
        params = {"format": _format}
        creating = True

        # request export of the dataset and wait for it to be ready
        while True:
            with session.get(url, params=params, stream=True) as response:
                response.raise_for_status()

                if response.status_code == 200:
                    if response.headers["Content-Type"] != "application/zip":
                        raise RequestException("Unexpected response content type")

                    # file is ready for download
                    # switch progress bar from 'waiting for' to 'downloading'
                    if progress is not None:
                        progress.reset()
                        progress.total = int(response.headers.get("Content-Length", 0))
                        progress.unit = "B"
                        progress.unit_scale = True
                        progress.unit_divisor = 1024
                        progress.set_description(f"Download dataset {id_}")

                    # download exported dataset
                    with open(output, "wb") as output_file:
                        for chunk in response.iter_content(chunk_size=4096):
                            output_file.write(chunk)
                            if progress is not None:
                                progress.update(len(chunk))
                    break

            if creating:
                params["action"] = "download"
                creating = False
                continue

            if progress is not None:
                progress.update(0)
            time.sleep(1)


def unzip_dataset(dataset):
    """Unzip a zip archive and remove the file"""
    output_dir = Path(dataset.stem)
    output_dir.mkdir()

    with zipfile.ZipFile(dataset) as archive:
        archive.extractall(path=output_dir)
        # members = archive.infolist()
        # for zipinfo in members:
        #    archive.extract(zipinfo, output_dir)

    os.unlink(dataset)


def _cvat_export_dataset_cli(
    cvat_params, id_, dataset_format, position, class_="task", unzip=True
):
    """Download a dataset from CVAT (with progress bar)"""

    output = Path(f"{dataset_format}_{class_}_{id_}")
    output_zip = output.with_suffix(".zip")
    if output.exists():
        print(f"{output} already exists, skipping download")
        return

    with tqdm(
        desc=f"Exporting dataset for {class_} {id_}",
        position=position,
        leave=False,
    ) as pbar:
        try:
            cvat_export_dataset(
                cvat_params,
                id_,
                output_zip,
                dataset_format,
                class_=class_,
                progress=pbar,
            )

        except requests.exceptions.RequestException as exc:
            tqdm.write(f"Failed exporting dataset {id_}: {exc}")
            return

        if unzip:
            pbar.set_description(f"Unpacking {output_zip}")
            unzip_dataset(output_zip)
        tqdm.write(f"Downloaded {output}")


def main():
    """Download dataset from CVAT (CLI)"""
    parser = configargparse.ArgParser(default_config_files=["~/.opentpod-tools"])
    parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
    parser.add_argument("--url", required=True, help="base URL of CVAT installation")
    parser.add_argument("--username", help="CVAT login username")
    parser.add_argument("--password", help="CVAT login password")
    parser.add_argument(
        "--no-unzip", action="store_true", help="Do not unpack datasets after download"
    )
    parser.add_argument(
        "-f",
        "--format",
        default="datumaro",
        help=""" \
             dataset format [datumaro, coco, pascal, labelme, mask, mot, tfrecord, \
             yolo] (defaults to datumaro) \
        """,
    )
    parser.add_argument(
        "--project", action="store_true", help="Type of dataset (project/task/job)"
    )
    parser.add_argument("--task", action="store_true", help="default: task")
    parser.add_argument("--job", action="store_true")
    parser.add_argument(
        "id", type=int, nargs="+", help="project/task/job id of dataset to download"
    )
    args = parser.parse_args()

    _auth = (args.username, args.password) if args.username or args.password else None

    if args.project:
        class_ = "project"
    elif args.job:
        class_ = "job"
    else:  # args.task | default
        class_ = "task"

    export_dataset = partial(
        _cvat_export_dataset_cli,
        (args.url, _auth),
        dataset_format=args.format,
        class_=class_,
        unzip=not args.no_unzip,
    )

    with ThreadPoolExecutor() as pool:
        for position, id_ in enumerate(args.id, 1):
            pool.submit(export_dataset, id_, position=position)
    print()


if __name__ == "__main__":
    main()
