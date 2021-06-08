# SPDX-FileCopyrightText: 2020 Carnegie Mellon University
#
# SPDX-License-Identifier: Apache-2.0

from opentpod_tools import __version__


def test_version():
    assert __version__ == "0.1.0"
