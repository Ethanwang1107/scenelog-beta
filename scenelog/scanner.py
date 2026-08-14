"""素材扫描器 — 递归扫描目录，发现可处理素材"""

import logging
from pathlib import Path

from scenelog.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def scan_media(
    source_dir: Path,
    excluded_dirs: list[Path] | None = None,
) -> list[Path]:
    """递归扫描素材目录，返回支持的媒体文件列表（按文件名排序）。

    忽略隐藏文件（以 . 开头）和 _scenelog 目录。
    """
    files: list[Path] = []
    excluded = [source_dir / "_scenelog"]
    excluded.extend(excluded_dirs or [])

    for p in sorted(source_dir.rglob("*")):
        is_excluded = any(_is_under(p, directory) for directory in excluded)
        if p.is_file() and not _is_hidden(p) and not is_excluded:
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)

    logger.info("扫描完成：发现 %d 个素材文件", len(files))
    return files


def _is_hidden(path: Path) -> bool:
    """判断路径或其任一部分是否以 . 开头"""
    return any(part.startswith(".") for part in path.parts)


def _is_under(path: Path, parent: Path) -> bool:
    """判断 path 是否在 parent 目录下"""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
