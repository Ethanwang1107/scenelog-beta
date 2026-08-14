"""独立运行时的外部可执行文件定位。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_STANDARD_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
_TRAE_PATH_MARKERS = ("/trae", "/trae solo")


def find_executable(name: str) -> str | None:
    """查找用户配置或标准系统安装的可执行文件，忽略 Trae 内置路径。"""
    override = os.environ.get(f"SCENELOG_{name.upper().replace('-', '_')}_BIN")
    if override:
        path = Path(override).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return None

    for directory in _STANDARD_BIN_DIRS:
        path = Path(directory) / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    path = shutil.which(name)
    if path and not _is_trae_path(path):
        return path
    return None


def _is_trae_path(path: str) -> bool:
    normalized = path.lower()
    return any(marker in normalized for marker in _TRAE_PATH_MARKERS)
