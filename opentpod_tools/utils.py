# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
# SPDX-License-Identifier: Apache-2.0

import contextlib
from pathlib import Path

from datumaro.plugins.data_formats.datumaro.format import DatumaroPath

def datumaro_fixup(path: Path) -> None:
    """Datumaro datasets that were exported without copying the media files
    have an empty 'images/' or 'video/' subdirectory which causes problems
    during import because it will try to resolve all media paths relative
    to those directories."""
    for media_path in [DatumaroPath.IMAGES_DIR, DatumaroPath.VIDEO_DIR]:
        with contextlib.suppress(FileNotFoundError, OSError):
            path.joinpath(media_path).rmdir()
