#
#  Copyright (c) 2019-2020 Carnegie Mellon University
#  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for object detection providers"""

import os
import shutil
from pathlib import Path

import requests
from logzero import logger
from tqdm import tqdm

# Registry to track available tensorflow detectors
REGISTRY = {}

XDG_CACHE_DIR = (
    Path(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "opentpod-tools"
)


def get_cache_entry(entry_name):
    """returns path to entry in cache or None"""
    cache_dir = XDG_CACHE_DIR / entry_name
    if not cache_dir.exists():
        return None
    return cache_dir


def download_and_extract_url_tarball_to_cache_dir(tarball_url, entry_name):
    """Download and extract tarball from url to a cache_dir
    entry_name: used as subdir name within the cache_dir. no '/' allowed
    """
    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tarball_basename = tarball_url.split("/")[-1]
    download_path = XDG_CACHE_DIR / tarball_basename

    if not download_path.exists():
        with tqdm(
            desc=f"Downloading {tarball_basename}",
            total=float("inf"),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress, requests.get(tarball_url, stream=True) as response:
            response.raise_for_status()

            progress.total = int(response.headers.get("Content-Length", 0))

            try:
                with open(download_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=4096):
                        output_file.write(chunk)
                        progress.update(len(chunk))
            except IOError as exc:
                output_file.unlink()
                logger.exception(exc)
                raise exc

    logger.info("Extracting %s", download_path)
    output_dir = XDG_CACHE_DIR / entry_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # try to unpack, remove partially extracted cache state on failure
    # shutil.unpack_archive docs don't list all possible exception so we have to
    # use a generic catch-all.
    try:
        shutil.unpack_archive(os.fspath(download_path), os.fspath(output_dir))
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(exc)
        shutil.rmtree(output_dir)
