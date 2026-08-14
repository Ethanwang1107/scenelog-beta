"""环境预检 — 处理前检查运行环境"""

from __future__ import annotations

import json
import logging
import os
import shutil
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

from scenelog.config import (
    DISK_SAFE_MIN_GB,
    VISION_VLM_MODEL,
    WHISPER_CPP_BINARY,
    WHISPER_CPP_MODEL,
)
from scenelog.dependencies import find_executable

logger = logging.getLogger(__name__)


class PreflightError(Exception):
    """预检失败"""


def run_preflight(
    source_dir: Path,
    output_dir: Path,
    vision_enabled: bool,
    people_enabled: bool = False,
) -> None:
    """执行环境预检，任一不通过则抛出 PreflightError。"""
    checks = [
        _check_source_readable(source_dir),
        _check_output_writable(output_dir),
        _check_ffmpeg(),
        _check_ffprobe(),
        _check_whisper(),
        _check_ollama(vision_enabled),
        _check_people_models(output_dir, people_enabled),
        _check_disk_space(output_dir),
        _check_no_concurrent(output_dir),
    ]

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise PreflightError("环境预检失败:\n  " + "\n  ".join(failed))

    logger.info("环境预检通过")


def _check_source_readable(source_dir: Path) -> tuple[bool, str]:
    if not source_dir.is_dir():
        return False, f"素材目录不存在: {source_dir}"
    if not os.access(str(source_dir), os.R_OK):
        return False, f"素材目录不可读: {source_dir}"
    return True, ""


def _check_output_writable(output_dir: Path) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(str(output_dir), os.W_OK):
        return False, f"输出目录不可写: {output_dir}"
    return True, ""


def _check_ffmpeg() -> tuple[bool, str]:
    if find_executable("ffmpeg") is None:
        return False, "未找到 ffmpeg，请安装: brew install ffmpeg"
    return True, ""


def _check_ffprobe() -> tuple[bool, str]:
    if find_executable("ffprobe") is None:
        return False, "未找到 ffprobe，请安装: brew install ffmpeg"
    return True, ""


def _check_whisper() -> tuple[bool, str]:
    binary = Path(WHISPER_CPP_BINARY).expanduser()
    model = Path(WHISPER_CPP_MODEL).expanduser()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return False, (
            "未找到 whisper-cli，请设置 SCENELOG_WHISPER_CPP_BIN，"
            "或按 INSTALL.md 安装 whisper.cpp"
        )
    if not model.is_file():
        return False, (
            "未找到 Whisper 模型，请设置 SCENELOG_WHISPER_CPP_MODEL，"
            "或按 INSTALL.md 下载模型"
        )
    return True, ""


def _check_ollama(vision_enabled: bool) -> tuple[bool, str]:
    if find_executable("ollama") is None:
        return False, "未找到 Ollama，请安装: brew install --cask ollama"

    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
            models = {
                item.get("name", "")
                for item in json.loads(response.read()).get("models", [])
            }
    except (URLError, TimeoutError, json.JSONDecodeError):
        return False, "Ollama 服务未运行，请执行: ollama serve"

    if not any(
        model == "qwen2.5" or model.startswith("qwen2.5:")
        for model in models
    ):
        return False, "缺少摘要模型，请执行: ollama pull qwen2.5"
    if vision_enabled:
        visual_models = (
            [VISION_VLM_MODEL]
            if VISION_VLM_MODEL
            else ["qwen2.5vl:3b", "qwen2.5vl:7b", "llava:7b", "llava"]
        )
        if not any(model in models for model in visual_models):
            return False, f"缺少画面模型，请执行: ollama pull {visual_models[0]}"
    return True, ""


def _check_people_models(
    output_dir: Path,
    people_enabled: bool,
) -> tuple[bool, str]:
    if not people_enabled:
        return True, ""
    from scenelog.people import PeopleStore, missing_models

    # 未登记关键人物时，处理阶段会直接跳过人物识别。
    # 注册命令自身会检查模型，因此这里无需阻止普通素材处理。
    if not PeopleStore(output_dir).has_registered_people():
        return True, ""

    missing = missing_models()
    if missing:
        return False, "缺少人物模型，请执行: scenelog people setup"
    return True, ""


def _check_disk_space(output_dir: Path) -> tuple[bool, str]:
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < DISK_SAFE_MIN_GB:
        return False, f"磁盘空间不足: {free_gb:.1f}GB < {DISK_SAFE_MIN_GB}GB (安全阈值)"
    return True, ""


def _check_no_concurrent(output_dir: Path) -> tuple[bool, str]:
    """简单锁：检查是否有运行中的任务标记"""
    lock_file = output_dir / ".running"
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                lock_file.unlink(missing_ok=True)
                return True, ""
            except PermissionError:
                pass
            return False, "检测到另一个 scenelog 任务正在运行中"
        lock_file.unlink(missing_ok=True)
    return True, ""
