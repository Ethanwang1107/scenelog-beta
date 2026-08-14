"""material_id 生成 — 素材稳定唯一标识"""

import hashlib
from pathlib import Path

from scenelog.config import PIPELINE_VERSION


def generate_material_id(source_dir: Path, file_path: Path) -> str:
    """为素材生成稳定的 material_id。

    material_id 只代表素材在当前素材根目录中的稳定身份。文件内容是否变化
    由 source_fingerprint 单独判断，避免内容变化后旧状态和旧索引失联。
    """
    rel_path = file_path.resolve().relative_to(source_dir.resolve())
    identity = rel_path.as_posix()
    hash_hex = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    return f"mat_{hash_hex}"


def compute_source_fingerprint(source_dir: Path, file_path: Path) -> str:
    """计算源素材快速指纹，用于判断内容是否变化。

    返回: "rel_path|size|mtime|head_hash"
    """
    rel_path = file_path.resolve().relative_to(source_dir.resolve())
    stat = file_path.stat()

    try:
        with open(file_path, "rb") as f:
            head = f.read(65536)
    except OSError:
        head = b""

    head_hash = hashlib.sha256(head).hexdigest()[:16]
    return f"{rel_path}|{stat.st_size}|{int(stat.st_mtime)}|{head_hash}"


def material_meta(source_dir: Path, file_path: Path) -> dict:
    """返回素材元信息字典，供 Excel 和状态管理使用。"""
    rel_path = file_path.resolve().relative_to(source_dir.resolve())
    stat = file_path.stat()

    return {
        "material_id": generate_material_id(source_dir, file_path),
        "file_name": file_path.name,
        "rel_path": str(rel_path),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "fingerprint": compute_source_fingerprint(source_dir, file_path),
        "pipeline_version": PIPELINE_VERSION,
    }
