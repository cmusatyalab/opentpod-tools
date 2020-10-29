#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
"""Download datasets from one or more CVAT tasks.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import time
import zipfile

import configargparse
import requests
from tqdm.auto import tqdm


def cvat_export_dataset(cvat_params, task_id, output, dataset_format, progress=None):
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

        url = f"{cvat_url}/api/v1/tasks/{task_id}/dataset"
        params = {"format": _format}
        creating = True

        # request export of the dataset and wait for it to be ready
        while True:
            with session.get(url, params=params, stream=True) as response:
                response.raise_for_status()

                if response.status_code == 200:
                    # file is ready for download
                    # switch progress bar from 'waiting for' to 'downloading'
                    if progress is not None:
                        progress.reset()
                        progress.total = int(response.headers.get("Content-Length", 0))
                        progress.unit = "B"
                        progress.unit_scale = True
                        progress.unit_divisor = 1024
                        progress.set_description(f"Download dataset {task_id}")

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
    cvat_params, task_id, dataset_format, position, unzip=True
):
    """Download a dataset from CVAT (with progress bar)"""

    output = Path(f"{dataset_format}_{task_id}")
    output_zip = output.with_suffix(".zip")
    if output.exists():
        print(f"{output} already exists, skipping download")
        return

    with tqdm(
        desc=f"Exporting dataset for task {task_id}",
        position=position,
        leave=False,
    ) as pbar:
        try:
            cvat_export_dataset(
                cvat_params, task_id, output_zip, dataset_format, progress=pbar
            )

        except requests.exceptions.HTTPError as exc:
            tqdm.write("Failed exporting dataset {}: {}".format(task_id, exc))

        if unzip:
            tqdm.write(f"Unpacking {output_zip}")
            unzip_dataset(output_zip)


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
        "--format",
        default="datumaro",
        help=""" \
             dataset format [datumaro, coco, pascal, labelme, mask, mot, tfrecord, \
             yolo] (defaults to datumaro) \
        """,
    )
    parser.add_argument(
        "task_id", type=int, nargs="+", help="task id of dataset to download"
    )
    args = parser.parse_args()

    _auth = (args.username, args.password) if args.username or args.password else None

    export_dataset = partial(
        _cvat_export_dataset_cli, (args.url, _auth), dataset_format=args.format
    )

    with ThreadPoolExecutor() as pool:
        for position, task in enumerate(args.task_id, 1):
            pool.submit(
                export_dataset, task, position=position, unzip=not args.no_unzip
            )


if __name__ == "__main__":
    main()
