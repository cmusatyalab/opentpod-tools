# SPDX-FileCopyrightText: 2020-2021 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

from . import tfod_faster_rcnn, tfod_ssd
from .utils import REGISTRY

__all__ = ["REGISTRY", "tfod_faster_rcnn", "tfod_ssd"]
